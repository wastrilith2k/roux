/* Observation Dashboard — Conversation-scoped view */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var activeTab = 'facts';
    var tabCompanion = null;
    var conversations = [];          // Full list from API
    var activeGroup = null;          // Currently selected group {key, participants, convIds, ...}
    var activeConvIds = [];          // Conversation IDs for the active group
    var activeParticipants = [];     // Derived from activeGroup (or all if none)
    var colorMap = {};               // name -> hex color (stable across views)
    var connected = false;
    var messageOffset = 0;

    var COLORS = ['#58a6ff', '#bc8cff', '#56d364', '#f0883e', '#f778ba', '#79c0ff', '#ffa657', '#ff7b72'];
    var feed = document.getElementById('feed');

    // -----------------------------------------------------------------------
    // URL query-param helpers (for shareable links)
    // -----------------------------------------------------------------------
    function getUrlParams() {
        var p = new URLSearchParams(window.location.search);
        return { conv: p.get('conv'), session: p.get('session') };
    }

    function setUrlParams(group, sessionId) {
        var p = new URLSearchParams();
        if (group) {
            p.set('conv', group.key);
            if (sessionId != null) p.set('session', sessionId);
        }
        var qs = p.toString();
        history.replaceState(null, '', window.location.pathname + (qs ? '?' + qs : ''));
    }

    // -----------------------------------------------------------------------
    // localStorage helpers
    // -----------------------------------------------------------------------
    var LS_KEY = 'observe_ui_v1';

    function lsGet() {
        try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch (e) { return {}; }
    }

    function lsSet(updates) {
        try {
            var s = lsGet();
            Object.assign(s, updates);
            localStorage.setItem(LS_KEY, JSON.stringify(s));
        } catch (e) {}
    }

    // -----------------------------------------------------------------------
    // SocketIO
    // -----------------------------------------------------------------------
    var socket = io('/observe', {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: 2000,
    });

    socket.on('connect', function () {
        connected = true;
        document.getElementById('status-dot').className = 'status-dot connected';
        document.getElementById('status-text').textContent = 'Connected';
        init();
    });

    socket.on('disconnect', function () {
        connected = false;
        document.getElementById('status-dot').className = 'status-dot';
        document.getElementById('status-text').textContent = 'Disconnected';
    });

    socket.on('sim:message', function (data) {
        var speaker = (data.speaker || '').toLowerCase();
        if (!activeGroup || activeParticipants.includes(speaker)) {
            appendMessage(data);
            autoScroll();
        }
    });

    socket.on('sim:day_start', function (data) {
        var el = document.createElement('div');
        el.className = 'day-marker';
        el.innerHTML = '<span>Day ' + data.day + ' — ' + (data.date_label || '') + '</span>';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:day_end', function (data) {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = '— End of Day ' + data.day + ' —';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:conversation_start', function (data) {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = data.initiator + ' → ' + data.responder +
            ' (' + data.num_exchanges + ' exchanges)';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:conversation_end', function () {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = '───';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:task_complete', function (data) {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = '✓ ' + (data.companion_id || '') + ': ' + (data.task_name || '') +
            (data.summary ? ' — ' + data.summary : '');
        feed.appendChild(el);
        autoScroll();
    });

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    function init() {
        fetch('/api/observe/conversations')
            .then(function (r) { return r.json(); })
            .then(function (list) {
                conversations = list;
                buildColorMap();
                buildConvTree();
                restoreSavedSelection();
                startPolling();
            })
            .catch(function () {
                conversations = [];
                buildConvTree();
                selectGroup(null);
                startPolling();
            });
    }

    function restoreSavedSelection() {
        var groups = window._convGroups || {};
        var restored = false;

        // URL params take priority — enables shareable links
        var url = getUrlParams();
        if (url.conv && groups[url.conv]) {
            var g = groups[url.conv];
            if (url.session) {
                var sid = parseInt(url.session, 10);
                if (g.convIds.indexOf(sid) !== -1) {
                    selectSession(g, sid);
                    restored = true;
                }
            }
            if (!restored) {
                selectGroup(g);
                restored = true;
            }
        }

        // Fall back to localStorage
        if (!restored) {
            var saved = lsGet();
            if (saved.convGroupKey && groups[saved.convGroupKey]) {
                var sg = groups[saved.convGroupKey];
                if (saved.convSessionId && sg.convIds.indexOf(saved.convSessionId) !== -1) {
                    selectSession(sg, saved.convSessionId);
                } else {
                    selectGroup(sg);
                }
                restored = true;
            }
        }

        if (!restored) selectGroup(null);

        // Restore active tab
        var lsSaved = lsGet();
        if (lsSaved.activeTab) {
            var tabBtn = document.querySelector('.tab-bar button[data-tab="' + lsSaved.activeTab + '"]');
            if (tabBtn) tabBtn.click();
        }
    }

    // -----------------------------------------------------------------------
    // Stable color map — alphabetical assignment so colors never shift
    // -----------------------------------------------------------------------
    function buildColorMap() {
        var all = new Set();
        conversations.forEach(function (c) {
            (c.participants || []).forEach(function (p) { all.add(p); });
        });
        var sorted = Array.from(all).sort();
        sorted.forEach(function (name, idx) {
            colorMap[name] = COLORS[idx % COLORS.length];
        });
    }

    function getParticipantColor(name) {
        if (colorMap[name]) return colorMap[name];
        var h = 0;
        for (var i = 0; i < name.length; i++) h = ((h << 5) - h) + name.charCodeAt(i);
        return COLORS[Math.abs(h) % COLORS.length];
    }

    // -----------------------------------------------------------------------
    // Conversation tree picker
    // -----------------------------------------------------------------------
    function buildConvTree() {
        // Build groups (participant-set keyed)
        var groups = {};
        conversations.forEach(function (conv) {
            var key = (conv.participants || []).slice().sort().join(',');
            if (!groups[key]) {
                groups[key] = {
                    key: key,
                    participants: (conv.participants || []).slice().sort(),
                    convIds: [],
                    sessions: [],
                    messageCount: 0,
                    latestDate: null,
                };
            }
            groups[key].convIds.push(conv.id);
            groups[key].sessions.push({ id: conv.id, date: conv.date, count: conv.message_count });
            groups[key].messageCount += conv.message_count;
            if (!groups[key].latestDate || conv.date > groups[key].latestDate) {
                groups[key].latestDate = conv.date;
            }
        });
        window._convGroups = groups;

        var saved = lsGet();
        var expandedKeys = saved.groupExpanded || {};

        var tree = document.getElementById('conv-tree');
        tree.innerHTML = '';

        // "All conversations" row
        var allRow = document.createElement('div');
        allRow.className = 'conv-all-row';
        allRow.textContent = 'All conversations';
        allRow.addEventListener('click', function () {
            selectGroup(null);
            closeConvDropdown();
        });
        tree.appendChild(allRow);

        // Sort groups: larger participant sets first, then alpha
        var sortedKeys = Object.keys(groups).sort(function (a, b) {
            var pa = groups[a].participants.length, pb = groups[b].participants.length;
            if (pa !== pb) return pb - pa;
            return a.localeCompare(b);
        });

        sortedKeys.forEach(function (key) {
            var g = groups[key];
            // Default expanded unless explicitly saved as false
            var isExpanded = expandedKeys[key] !== false;

            var groupEl = document.createElement('div');
            groupEl.className = 'conv-group';
            groupEl.dataset.key = key;

            var header = document.createElement('div');
            header.className = 'conv-group-header';

            var arrow = document.createElement('span');
            arrow.className = 'conv-expand-arrow' + (isExpanded ? ' open' : '');
            arrow.textContent = '▶';

            var nameEl = document.createElement('span');
            nameEl.className = 'conv-group-name';
            nameEl.textContent = g.participants.map(capitalize).join(' & ');

            var metaEl = document.createElement('span');
            metaEl.className = 'conv-group-meta';
            metaEl.textContent = g.sessions.length + ' · ' + g.messageCount + ' msgs';

            var sessionsEl = document.createElement('div');
            sessionsEl.className = 'conv-sessions' + (isExpanded ? ' open' : '');

            // Sort sessions by date
            var sortedSessions = g.sessions.slice().sort(function (a, b) {
                return (a.date || '').localeCompare(b.date || '');
            });

            sortedSessions.forEach(function (sess) {
                var sessEl = document.createElement('div');
                sessEl.className = 'conv-session';
                sessEl.dataset.convId = sess.id;
                sessEl.textContent = (sess.date || 'Unknown') + ' · ' + sess.count + ' msgs';
                sessEl.addEventListener('click', function (e) {
                    e.stopPropagation();
                    selectSession(g, sess.id);
                    closeConvDropdown();
                });
                sessionsEl.appendChild(sessEl);
            });

            // Arrow click = expand/collapse only (don't select)
            arrow.addEventListener('click', function (e) {
                e.stopPropagation();
                isExpanded = !isExpanded;
                arrow.classList.toggle('open', isExpanded);
                sessionsEl.classList.toggle('open', isExpanded);
                var exp = lsGet().groupExpanded || {};
                exp[key] = isExpanded;
                lsSet({ groupExpanded: exp });
            });

            // Header click = select whole group
            header.addEventListener('click', function () {
                selectGroup(g);
                closeConvDropdown();
            });

            header.appendChild(arrow);
            header.appendChild(nameEl);
            header.appendChild(metaEl);
            groupEl.appendChild(header);
            groupEl.appendChild(sessionsEl);
            tree.appendChild(groupEl);
        });

        // Update active highlights whenever the tree is rebuilt
        updateTreeActive();

        // Wire up picker button
        var btn = document.getElementById('conv-picker-btn');
        btn.onclick = function (e) {
            e.stopPropagation();
            var dd = document.getElementById('conv-dropdown');
            var opening = !dd.classList.contains('open');
            dd.classList.toggle('open', opening);
            btn.classList.toggle('open', opening);
            if (opening) document.getElementById('conv-search').focus();
        };

        // Search/filter
        document.getElementById('conv-search').addEventListener('input', function () {
            filterConvTree(this.value.toLowerCase().trim());
        });
    }

    function updateTreeActive() {
        // Highlight the active group / session in the tree
        var groupKey = activeGroup ? activeGroup.key : null;
        var sessionId = activeGroup && activeGroup.sessionId ? activeGroup.sessionId : null;

        document.querySelectorAll('.conv-all-row').forEach(function (el) {
            el.classList.toggle('active', !activeGroup);
        });
        document.querySelectorAll('.conv-group-header').forEach(function (el) {
            var gEl = el.closest('.conv-group');
            el.classList.toggle('active', !sessionId && gEl && gEl.dataset.key === groupKey);
        });
        document.querySelectorAll('.conv-session').forEach(function (el) {
            el.classList.toggle('active', sessionId != null && el.dataset.convId == sessionId);
        });
    }

    function updatePickerLabel() {
        var label = document.querySelector('.conv-picker-label');
        if (!label) return;
        if (!activeGroup) {
            label.textContent = 'All conversations';
        } else {
            var names = activeGroup.participants.map(capitalize).join(' & ');
            if (activeGroup.sessionId) {
                label.textContent = names + ' · 1 session';
            } else {
                label.textContent = names;
            }
        }
    }

    function filterConvTree(query) {
        var allRow = document.querySelector('.conv-all-row');
        if (allRow) allRow.style.display = !query ? '' : 'none';
        document.querySelectorAll('.conv-group').forEach(function (el) {
            var name = (el.querySelector('.conv-group-name') || {}).textContent || '';
            el.style.display = name.toLowerCase().includes(query) ? '' : 'none';
        });
    }

    function closeConvDropdown() {
        var dd = document.getElementById('conv-dropdown');
        var btn = document.getElementById('conv-picker-btn');
        if (dd) dd.classList.remove('open');
        if (btn) btn.classList.remove('open');
    }

    // Close dropdown when clicking outside
    document.addEventListener('click', function (e) {
        var picker = document.getElementById('conv-picker');
        if (picker && !picker.contains(e.target)) closeConvDropdown();
    });

    // -----------------------------------------------------------------------
    // Group / session selection
    // -----------------------------------------------------------------------
    function selectGroup(group) {
        activeGroup = group;
        activeConvIds = group ? group.convIds.slice() : [];
        activeParticipants = group ? group.participants.slice() : getAllParticipants();
        tabCompanion = activeParticipants[0] || null;

        setUrlParams(group, null);
        lsSet({
            convGroupKey: group ? group.key : null,
            convSessionId: null,
        });

        updateTreeActive();
        updatePickerLabel();
        buildParticipantPills();
        buildStatePanels();
        buildTabToggles();
        messageOffset = 0;
        feed.innerHTML = '';
        loadAllMessages();
        pollActiveTab();
    }

    function selectSession(group, convId) {
        // Synthetic group representing a single session within the parent group
        activeGroup = {
            key: group.key,
            participants: group.participants,
            convIds: [convId],
            sessionId: convId,
            messageCount: ((group.sessions || []).find(function (s) { return s.id === convId; }) || {}).count || 0,
        };
        activeConvIds = [convId];
        activeParticipants = group.participants.slice();
        tabCompanion = activeParticipants[0] || null;

        setUrlParams(group, convId);
        lsSet({
            convGroupKey: group.key,
            convSessionId: convId,
        });

        updateTreeActive();
        updatePickerLabel();
        buildParticipantPills();
        buildStatePanels();
        buildTabToggles();
        messageOffset = 0;
        feed.innerHTML = '';
        loadAllMessages();
        pollActiveTab();
    }

    function getAllParticipants() {
        var seen = new Set();
        conversations.forEach(function (c) {
            (c.participants || []).forEach(function (p) { seen.add(p); });
        });
        return Array.from(seen).sort();
    }

    // -----------------------------------------------------------------------
    // Participant pills (header bar)
    // -----------------------------------------------------------------------
    function buildParticipantPills() {
        var container = document.getElementById('selector-buttons');
        container.innerHTML = '';
        activeParticipants.forEach(function (cid) {
            var color = getParticipantColor(cid);
            var pill = document.createElement('span');
            pill.textContent = capitalize(cid);
            pill.style.cssText = 'display:inline-block;padding:2px 10px;border-radius:12px;border:1px solid ' +
                color + ';color:' + color + ';font-size:12px;margin-right:6px;';
            container.appendChild(pill);
        });
    }

    // -----------------------------------------------------------------------
    // State panels (left sidebar) — with localStorage collapse persistence
    // -----------------------------------------------------------------------
    function buildStatePanels() {
        var container = document.getElementById('state-panels');
        container.innerHTML = '';
        var savedCollapsed = lsGet().panelCollapsed || {};

        activeParticipants.forEach(function (cid) {
            var color = getParticipantColor(cid);
            var panel = document.createElement('div');
            panel.className = 'side-panel';
            panel.id = 'panel-' + cid;

            var titleBar = document.createElement('div');
            titleBar.className = 'panel-title';
            titleBar.style.color = color;
            titleBar.style.cursor = 'pointer';

            var isCollapsed = !!savedCollapsed[cid];
            titleBar.innerHTML = '<span class="collapse-arrow">' + (isCollapsed ? '&#9654;' : '&#9660;') + '</span> ' + capitalize(cid);

            var body = document.createElement('div');
            body.className = 'panel-body';
            body.id = 'panel-body-' + cid;
            if (isCollapsed) body.style.display = 'none';

            body.innerHTML =
                '<div class="state-card"><h3>Mood</h3><div class="state-value"><span class="mood-tag" id="' + cid + '-mood">--</span></div></div>' +
                '<div class="state-card"><h3>Energy</h3><div class="state-value" id="' + cid + '-energy-label">--</div>' +
                '<div class="bar-container"><div class="bar-fill energy" id="' + cid + '-energy-bar" style="width:0%"></div></div></div>' +
                '<div class="state-card"><h3>Mode</h3><div class="state-value" id="' + cid + '-mode">--</div></div>' +
                '<div class="state-card"><h3>Scene</h3><div class="state-value" id="' + cid + '-scene">--</div></div>' +
                '<div class="state-card editable"><h3>Closeness</h3><div class="state-value" id="' + cid + '-closeness-label">--</div>' +
                '<div class="bar-container"><div class="bar-fill closeness" id="' + cid + '-closeness-bar" style="width:0%"></div></div>' +
                '<input type="range" class="state-slider" id="' + cid + '-closeness-slider" min="0" max="100" value="0" style="display:none">' +
                '</div>' +
                '<div class="state-card editable"><h3>Emotion Profile</h3><div class="state-value" id="' + cid + '-emotion">--</div>' +
                '<input type="text" class="state-input" id="' + cid + '-emotion-input" placeholder="e.g. Guarded, Open, Trusting" style="display:none">' +
                '</div>' +
                '<div class="state-card editable"><h3>Romance</h3><div class="state-value" id="' + cid + '-romance">--</div>' +
                '<input type="range" class="state-slider" id="' + cid + '-romance-slider" min="0" max="100" value="0" style="display:none">' +
                '</div>' +
                '<button class="edit-state-btn" id="' + cid + '-edit-btn">Edit</button>' +
                '<div class="edit-actions" id="' + cid + '-edit-actions" style="display:none">' +
                '<button class="save-state-btn" onclick="saveState(\'' + cid + '\')">Save</button>' +
                '<button class="cancel-state-btn" onclick="cancelEdit(\'' + cid + '\')">Cancel</button>' +
                '</div>';

            (function (companionId) {
                body.querySelector('#' + companionId + '-edit-btn').addEventListener('click', function () {
                    toggleEdit(companionId);
                });
            })(cid);

            titleBar.addEventListener('click', function () {
                var nowCollapsed = body.style.display !== 'none';
                body.style.display = nowCollapsed ? 'none' : '';
                titleBar.querySelector('.collapse-arrow').innerHTML = nowCollapsed ? '&#9654;' : '&#9660;';
                // Persist collapse state
                var pc = lsGet().panelCollapsed || {};
                pc[cid] = nowCollapsed;
                lsSet({ panelCollapsed: pc });
            });

            panel.appendChild(titleBar);
            panel.appendChild(body);
            container.appendChild(panel);
        });
    }

    function buildTabToggles() {
        var container = document.getElementById('tab-companion-toggle');
        container.innerHTML = '';
        var saved = lsGet();
        // Restore tabCompanion from localStorage if it's in the current participant set
        if (saved.tabCompanion && activeParticipants.includes(saved.tabCompanion)) {
            tabCompanion = saved.tabCompanion;
        } else {
            tabCompanion = activeParticipants[0] || null;
        }

        activeParticipants.forEach(function (cid) {
            var color = getParticipantColor(cid);
            var btn = document.createElement('button');
            btn.textContent = capitalize(cid);
            btn.id = 'tab-cid-' + cid;
            btn.style.color = color;
            btn.className = cid === tabCompanion ? 'active-companion' : '';
            btn.addEventListener('click', function () { setTabCompanion(cid); });
            container.appendChild(btn);
        });
    }

    // -----------------------------------------------------------------------
    // Message rendering
    // -----------------------------------------------------------------------
    function appendMessage(data) {
        var el = document.createElement('div');
        var speaker = (data.speaker || '').toLowerCase();
        var color = getParticipantColor(speaker);
        el.className = 'message';
        el.style.borderLeftColor = color;

        var html = '<div class="speaker" style="color:' + color + '">' + esc(data.speaker || speaker) + '</div>';
        html += '<div>' + esc(data.content || '') + '</div>';

        if (data.timestamp || data._ts) {
            var ts = data.timestamp || data._ts;
            try {
                var d = new Date(ts);
                html += '<div class="timestamp">' + d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + '</div>';
            } catch (e) { /* ignore */ }
        }

        el.innerHTML = html;
        feed.appendChild(el);
    }

    function loadAllMessages(append) {
        var LIMIT = 100;
        if (!append) {
            feed.innerHTML = '';
            messageOffset = 0;
        }
        var url = '/api/observe/messages?limit=' + LIMIT + '&offset=' + messageOffset;

        if (activeConvIds.length > 0) {
            url += '&conversation_ids=' + activeConvIds.join(',');
        }
        if (activeGroup && activeGroup.participants.length > 0) {
            activeGroup.participants.forEach(function (p) {
                url += '&companion_id=' + encodeURIComponent(p);
            });
        }

        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (msgs) {
                var lastDay = null;
                msgs.forEach(function (m) {
                    var day = m.timestamp ? m.timestamp.split('T')[0] : null;
                    if (day && day !== lastDay) {
                        var marker = document.createElement('div');
                        marker.className = 'day-marker';
                        marker.innerHTML = '<span>' + day + '</span>';
                        feed.appendChild(marker);
                        lastDay = day;
                    }
                    appendMessage({ speaker: m.speaker, content: m.content, timestamp: m.timestamp });
                });
                messageOffset += msgs.length;
                var loadMoreBtn = document.getElementById('btn-load-more');
                if (loadMoreBtn) {
                    loadMoreBtn.style.display = msgs.length >= LIMIT ? '' : 'none';
                }
                if (!append) autoScroll();
            })
            .catch(function () { /* silent */ });
    }

    (function () {
        var btn = document.getElementById('btn-load-more');
        if (btn) btn.addEventListener('click', function () { loadAllMessages(true); });
    }());

    function autoScroll() { feed.scrollTop = feed.scrollHeight; }

    // -----------------------------------------------------------------------
    // State panel updater
    // -----------------------------------------------------------------------
    function updateStatePanel(companionId) {
        fetch('/api/observe/state?companion_id=' + companionId)
            .then(function (r) { return r.json(); })
            .then(function (s) {
                var prefix = companionId;
                var el, bar;

                el = document.getElementById(prefix + '-mood');
                if (el) {
                    var moodVal = '--';
                    if (s.mood && typeof s.mood === 'object') {
                        moodVal = s.mood.primary || s.mood.label || JSON.stringify(s.mood);
                    } else if (s.mood) {
                        moodVal = String(s.mood);
                    }
                    el.textContent = moodVal;
                }

                var energy = 50;
                if (s.energy && typeof s.energy === 'object') {
                    energy = s.energy.level != null ? s.energy.level : (s.energy.value != null ? s.energy.value : 50);
                } else if (typeof s.energy === 'number') {
                    energy = s.energy;
                }
                if (energy <= 1) energy = Math.round(energy * 100);
                var energyPct = Math.max(0, Math.min(100, energy));
                el = document.getElementById(prefix + '-energy-label');
                if (el) el.textContent = energyPct + '%';
                bar = document.getElementById(prefix + '-energy-bar');
                if (bar) {
                    bar.style.width = energyPct + '%';
                    bar.className = 'bar-fill energy' + (energyPct < 30 ? ' low' : energyPct < 60 ? ' medium' : '');
                }

                el = document.getElementById(prefix + '-mode');
                if (el) el.textContent = s.mode || '--';

                el = document.getElementById(prefix + '-scene');
                if (el) {
                    var sceneVal = '--';
                    if (s.scene && typeof s.scene === 'object') {
                        sceneVal = s.scene.activity || s.scene.location || JSON.stringify(s.scene);
                    } else if (s.scene) {
                        sceneVal = String(s.scene);
                    }
                    if (sceneVal === '--' && s.internal_state && s.internal_state.was_doing) {
                        sceneVal = s.internal_state.was_doing;
                    }
                    el.textContent = sceneVal;
                }

                el = document.getElementById(prefix + '-closeness-label');
                if (el) el.textContent = s.closeness || 0;
                bar = document.getElementById(prefix + '-closeness-bar');
                if (bar) bar.style.width = Math.min(100, s.closeness || 0) + '%';

                el = document.getElementById(prefix + '-emotion');
                if (el) el.textContent = s.emotion_profile || '--';

                el = document.getElementById(prefix + '-romance');
                if (el) el.textContent = (s.romance_level != null ? s.romance_level : '--');
            })
            .catch(function () { /* silent */ });
    }

    function pollStates() {
        activeParticipants.forEach(function (cid) { updateStatePanel(cid); });
    }

    // -----------------------------------------------------------------------
    // Sim status
    // -----------------------------------------------------------------------
    function pollSimStatus() {
        fetch('/api/observe/status')
            .then(function (r) { return r.json(); })
            .then(function (s) {
                var clockEl = document.getElementById('sim-clock');
                if (s.clock_time) {
                    clockEl.textContent = 'Sim: ' + s.clock_time;
                } else if (s.day) {
                    clockEl.textContent = 'Day ' + s.day;
                }
                if (s.status === 'running') {
                    document.getElementById('status-dot').className = 'status-dot running';
                }
            })
            .catch(function () { /* silent */ });
    }

    // -----------------------------------------------------------------------
    // Tab data — with localStorage persistence
    // -----------------------------------------------------------------------
    function setTabCompanion(cid) {
        tabCompanion = cid;
        lsSet({ tabCompanion: cid });
        document.querySelectorAll('#tab-companion-toggle button').forEach(function (btn) {
            btn.className = btn.id === 'tab-cid-' + cid ? 'active-companion' : '';
        });
        pollActiveTab();
    }

    function pollActiveTab() {
        if (!tabCompanion) return;
        var cid = tabCompanion;
        switch (activeTab) {
            case 'facts':
                loadTabData('/api/observe/facts?companion_id=' + cid + '&limit=50', 'tab-facts', formatFacts);
                break;
            case 'opinions':
                loadTabData('/api/observe/opinions?companion_id=' + cid, 'tab-opinions', formatOpinions);
                break;
            case 'curiosity':
                loadTabData('/api/observe/curiosity?companion_id=' + cid, 'tab-curiosity', formatCuriosity);
                break;
            case 'goals':
                loadTabData('/api/observe/goals?companion_id=' + cid, 'tab-goals', formatGoals);
                break;
            case 'episodes':
                loadTabData('/api/observe/episodes?companion_id=' + cid + '&limit=10', 'tab-episodes', formatEpisodes);
                break;
            case 'relationship':
                loadTabData('/api/observe/relationship?companion_id=' + cid, 'tab-relationship', formatRelationship);
                break;
        }
    }

    function loadTabData(url, elId, formatter) {
        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (data) { document.getElementById(elId).innerHTML = formatter(data); })
            .catch(function () {});
    }

    function formatFacts(facts) {
        if (!facts || !facts.length) return '<div class="empty-state">No facts yet</div>';
        var html = '<table class="data-table"><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Confidence</th></tr>';
        facts.forEach(function (f) {
            html += '<tr><td>' + esc(f.subject) + '</td><td>' + esc(f.predicate) + '</td><td>' + esc(f.object) + '</td><td>' + confidenceBadge(f.confidence) + '</td></tr>';
        });
        return html + '</table>';
    }

    function formatOpinions(opinions) {
        if (!opinions || !opinions.length) return '<div class="empty-state">No opinions yet</div>';
        var html = '<table class="data-table"><tr><th>Topic</th><th>Opinion</th><th>Confidence</th></tr>';
        opinions.forEach(function (o) {
            html += '<tr><td>' + esc(o.topic) + '</td><td>' + esc(truncate(o.opinion, 120)) + '</td><td>' + confidenceBadge(o.confidence) + '</td></tr>';
        });
        return html + '</table>';
    }

    function formatCuriosity(threads) {
        if (!threads || !threads.length) return '<div class="empty-state">No curiosity threads yet</div>';
        var html = '<table class="data-table"><tr><th>Topic</th><th>Urgency</th></tr>';
        threads.forEach(function (t) {
            html += '<tr><td>' + esc(t.topic || '') + '</td><td>' + confidenceBadge(t.urgency) + '</td></tr>';
        });
        return html + '</table>';
    }

    function formatGoals(goals) {
        if (!goals || !goals.length) return '<div class="empty-state">No goals yet</div>';
        var html = '<table class="data-table"><tr><th>Goal</th><th>Progress</th></tr>';
        goals.forEach(function (g) {
            html += '<tr><td>' + esc(g.goal) + '</td><td>' + Math.round((g.progress || 0) * 100) + '%</td></tr>';
        });
        return html + '</table>';
    }

    function formatEpisodes(episodes) {
        if (!episodes || !episodes.length) return '<div class="empty-state">No episodes yet</div>';
        var html = '<table class="data-table"><tr><th>When</th><th>Summary</th></tr>';
        episodes.forEach(function (e) {
            html += '<tr><td>' + shortDate(e.started_at) + '</td><td>' + esc(truncate(e.summary || '', 150)) + '</td></tr>';
        });
        return html + '</table>';
    }

    function formatRelationship(data) {
        if (!data || !data.evaluation) return '<div class="empty-state">No relationship data yet</div>';
        var html = '<div style="padding:8px;">';
        var eval_ = data.evaluation;
        if (typeof eval_ === 'object') {
            for (var key in eval_) {
                html += '<div style="margin-bottom:8px;"><strong>' + esc(key) + ':</strong> ' + esc(String(eval_[key])) + '</div>';
            }
        } else {
            html += esc(String(eval_));
        }
        return html + '</div>';
    }

    // -----------------------------------------------------------------------
    // Tab switching — persists active tab to localStorage
    // -----------------------------------------------------------------------
    document.querySelectorAll('.tab-bar button').forEach(function (btn) {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.tab-bar button').forEach(function (b) { b.classList.remove('active'); });
            document.querySelectorAll('.tab-content').forEach(function (c) { c.classList.remove('active'); });
            btn.classList.add('active');
            activeTab = btn.dataset.tab;
            lsSet({ activeTab: activeTab });
            document.getElementById('tab-' + activeTab).classList.add('active');
            pollActiveTab();
        });
    });

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    function esc(s) {
        if (!s) return '';
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }

    function truncate(s, n) {
        if (!s) return '';
        return s.length > n ? s.substring(0, n) + '...' : s;
    }

    function shortDate(iso) {
        if (!iso) return '--';
        try {
            var d = new Date(iso);
            return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (e) { return iso; }
    }

    function confidenceBadge(val) {
        if (val == null) return '--';
        var n = parseFloat(val);
        var cls = n >= 0.7 ? 'high' : n >= 0.4 ? 'medium' : 'low';
        return '<span class="confidence-badge ' + cls + '">' + n.toFixed(2) + '</span>';
    }

    // -----------------------------------------------------------------------
    // Polling
    // -----------------------------------------------------------------------
    function startPolling() {
        pollStates();
        pollSimStatus();
        pollActiveTab();
        setInterval(function () { pollStates(); pollSimStatus(); }, 5000);
        setInterval(pollActiveTab, 15000);
    }

    // -----------------------------------------------------------------------
    // State editing
    // -----------------------------------------------------------------------
    function toggleEdit(cid) {
        var editing = document.getElementById(cid + '-closeness-slider').style.display !== 'none';
        if (!editing) {
            var closenessEl = document.getElementById(cid + '-closeness-label');
            var emotionEl = document.getElementById(cid + '-emotion');
            var romanceEl = document.getElementById(cid + '-romance');

            var closenessSlider = document.getElementById(cid + '-closeness-slider');
            closenessSlider.value = parseInt(closenessEl.textContent) || 0;
            closenessSlider.style.display = 'block';
            closenessSlider.oninput = function () { closenessEl.textContent = this.value; };

            var romanceSlider = document.getElementById(cid + '-romance-slider');
            romanceSlider.value = Math.round(parseFloat(romanceEl.textContent) || 0);
            romanceSlider.style.display = 'block';
            romanceSlider.oninput = function () { romanceEl.textContent = this.value; };

            var emotionInput = document.getElementById(cid + '-emotion-input');
            emotionInput.value = emotionEl.textContent !== '--' ? emotionEl.textContent : '';
            emotionInput.style.display = 'block';

            document.getElementById(cid + '-edit-btn').style.display = 'none';
            document.getElementById(cid + '-edit-actions').style.display = 'flex';
        }
    }

    window.cancelEdit = function (cid) {
        document.getElementById(cid + '-closeness-slider').style.display = 'none';
        document.getElementById(cid + '-romance-slider').style.display = 'none';
        document.getElementById(cid + '-emotion-input').style.display = 'none';
        document.getElementById(cid + '-edit-btn').style.display = '';
        document.getElementById(cid + '-edit-actions').style.display = 'none';
        updateStatePanel(cid);
    };

    window.saveState = function (cid) {
        var closeness = parseInt(document.getElementById(cid + '-closeness-slider').value);
        var romance = parseInt(document.getElementById(cid + '-romance-slider').value);
        var emotion = document.getElementById(cid + '-emotion-input').value.trim();

        var payload = { companion_id: cid };
        if (!isNaN(closeness)) payload.closeness_score = closeness;
        if (!isNaN(romance)) payload.romance_level = romance;
        if (emotion) payload.emotion_profile = emotion;

        fetch('/api/observe/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.success) { window.cancelEdit(cid); }
                else { alert('Failed to save: ' + (data.error || 'unknown')); }
            })
            .catch(function (e) { alert('Save failed: ' + e.message); });
    };

    if (typeof io === 'undefined') {
        setTimeout(init, 500);
    }

}());
