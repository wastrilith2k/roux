/* Observation Dashboard — Dynamic character support */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let activeTab = 'facts';
    let tabCompanion = null;       // Set once companions load
    let activeFilters = new Set();  // Empty = show all, or set of companion_ids
    let companions = [];           // List of companion IDs from API
    let connected = false;
    let messageOffset = 0;         // Offset for paginated message loading

    // Color palette for dynamic characters
    const COLORS = ['#58a6ff', '#bc8cff', '#56d364', '#f0883e', '#f778ba', '#79c0ff', '#ffa657', '#ff7b72'];

    const feed = document.getElementById('feed');

    // -----------------------------------------------------------------------
    // SocketIO
    // -----------------------------------------------------------------------
    const socket = io('/observe', {
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

    // -----------------------------------------------------------------------
    // Simulation events (real-time)
    // -----------------------------------------------------------------------
    socket.on('sim:message', function (data) {
        if (!activeFilters.size || activeFilters.has((data.speaker || '').toLowerCase()) || activeFilters.has((data.listener || '').toLowerCase())) {
            appendMessage(data);
            autoScroll();
        }
    });

    socket.on('sim:day_start', function (data) {
        var el = document.createElement('div');
        el.className = 'day-marker';
        el.innerHTML = '<span>Day ' + data.day + ' \u2014 ' + (data.date_label || '') + '</span>';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:day_end', function (data) {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = '\u2014 End of Day ' + data.day + ' \u2014';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:conversation_start', function (data) {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = data.initiator + ' \u2192 ' + data.responder +
            ' (' + data.num_exchanges + ' exchanges)';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:conversation_end', function () {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = '\u2500\u2500\u2500';
        feed.appendChild(el);
        autoScroll();
    });

    socket.on('sim:task_complete', function (data) {
        var el = document.createElement('div');
        el.className = 'conversation-marker';
        el.textContent = '\u2713 ' + (data.companion_id || '') + ': ' + (data.task_name || '') +
            (data.summary ? ' \u2014 ' + data.summary : '');
        feed.appendChild(el);
        autoScroll();
    });

    // -----------------------------------------------------------------------
    // Init: fetch companions and build UI
    // -----------------------------------------------------------------------
    function init() {
        fetch('/api/observe/companions')
            .then(function (r) { return r.json(); })
            .then(function (list) {
                companions = list;
                if (!companions.length) {
                    companions = ['kai', 'mira']; // Fallback
                }
                tabCompanion = companions[0];
                buildCharacterSelector();
                buildStatePanels();
                buildTabToggles();
                populateCompanionSelect();
                loadAllMessages();
                startPolling();
            })
            .catch(function () {
                companions = ['kai', 'mira'];
                tabCompanion = 'kai';
                buildCharacterSelector();
                buildStatePanels();
                buildTabToggles();
                populateCompanionSelect();
                loadAllMessages();
                startPolling();
            });
    }

    function populateCompanionSelect() {
        var sel = document.getElementById('companion-select');
        if (!sel) return;
        // Keep the default "All" option and add one per companion
        sel.innerHTML = '<option value="">All</option>';
        companions.forEach(function (cid) {
            var opt = document.createElement('option');
            opt.value = cid;
            opt.textContent = capitalize(cid);
            sel.appendChild(opt);
        });
        sel.addEventListener('change', function () {
            messageOffset = 0;
            feed.innerHTML = '';
            loadAllMessages();
        });
    }

    // -----------------------------------------------------------------------
    // Build dynamic UI
    // -----------------------------------------------------------------------
    function getColor(idx) {
        return COLORS[idx % COLORS.length];
    }

    function buildCharacterSelector() {
        var container = document.getElementById('selector-buttons');
        container.innerHTML = '';
        companions.forEach(function (cid, idx) {
            var btn = document.createElement('button');
            btn.textContent = capitalize(cid);
            btn.className = 'selector-btn';
            btn.dataset.cid = cid;
            btn.dataset.color = getColor(idx);
            btn.style.borderColor = getColor(idx);
            btn.style.color = getColor(idx);
            btn.addEventListener('click', function () { filterByCharacter(cid); });
            container.appendChild(btn);
        });
    }

    function buildStatePanels() {
        var container = document.getElementById('state-panels');
        container.innerHTML = '';
        companions.forEach(function (cid, idx) {
            var panel = document.createElement('div');
            panel.className = 'side-panel';
            panel.id = 'panel-' + cid;

            var titleBar = document.createElement('div');
            titleBar.className = 'panel-title';
            titleBar.style.color = getColor(idx);
            titleBar.style.cursor = 'pointer';
            titleBar.innerHTML = '<span class="collapse-arrow">&#9660;</span> ' + capitalize(cid);

            var body = document.createElement('div');
            body.className = 'panel-body';
            body.id = 'panel-body-' + cid;
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

            // Wire up edit button
            (function(companionId) {
                body.querySelector('#' + companionId + '-edit-btn').addEventListener('click', function() {
                    toggleEdit(companionId);
                });
            })(cid);

            titleBar.addEventListener('click', function () {
                var isHidden = body.style.display === 'none';
                body.style.display = isHidden ? '' : 'none';
                titleBar.querySelector('.collapse-arrow').innerHTML = isHidden ? '&#9660;' : '&#9654;';
            });

            panel.appendChild(titleBar);
            panel.appendChild(body);
            container.appendChild(panel);
        });
    }

    function buildTabToggles() {
        var container = document.getElementById('tab-companion-toggle');
        container.innerHTML = '';
        companions.forEach(function (cid, idx) {
            var btn = document.createElement('button');
            btn.textContent = capitalize(cid);
            btn.id = 'tab-cid-' + cid;
            btn.style.color = getColor(idx);
            btn.className = cid === tabCompanion ? 'active-companion' : '';
            btn.addEventListener('click', function () { setTabCompanion(cid); });
            container.appendChild(btn);
        });
    }

    // -----------------------------------------------------------------------
    // Character filtering
    // -----------------------------------------------------------------------
    function filterByCharacter(cid) {
        // Toggle this character in the filter set
        if (activeFilters.has(cid)) {
            activeFilters.delete(cid);
        } else {
            activeFilters.add(cid);
        }
        updateSelectorVisuals();
        messageOffset = 0;
        loadAllMessages();
    }

    window.showAllCharacters = function () {
        activeFilters.clear();
        updateSelectorVisuals();
        messageOffset = 0;
        loadAllMessages();
    };

    function updateSelectorVisuals() {
        document.querySelectorAll('.selector-btn').forEach(function (btn) {
            var isActive = activeFilters.has(btn.dataset.cid);
            var color = btn.dataset.color;
            btn.classList.toggle('active', isActive);
            if (isActive) {
                btn.style.backgroundColor = color;
                btn.style.color = '#0d1117';
                btn.style.borderColor = color;
                btn.style.fontWeight = '700';
                btn.style.boxShadow = '0 0 8px ' + color + '66';
            } else {
                btn.style.backgroundColor = 'transparent';
                btn.style.color = color;
                btn.style.borderColor = color;
                btn.style.fontWeight = '';
                btn.style.boxShadow = '';
            }
        });
        var allBtn = document.getElementById('btn-show-all');
        var noneSelected = !activeFilters.size;
        allBtn.classList.toggle('active', noneSelected);
        if (noneSelected) {
            allBtn.style.backgroundColor = 'var(--bg-tertiary)';
            allBtn.style.color = 'var(--text-primary)';
            allBtn.style.fontWeight = '700';
        } else {
            allBtn.style.backgroundColor = 'transparent';
            allBtn.style.color = 'var(--text-secondary)';
            allBtn.style.fontWeight = '';
        }
        // Update count badge
        if (activeFilters.size > 0 && activeFilters.size < companions.length) {
            allBtn.textContent = 'Show All (' + activeFilters.size + ' selected)';
        } else {
            allBtn.textContent = 'Show All';
        }
    }

    // -----------------------------------------------------------------------
    // Message rendering
    // -----------------------------------------------------------------------
    function appendMessage(data) {
        var el = document.createElement('div');
        var speaker = (data.speaker || '').toLowerCase();
        var idx = companions.indexOf(speaker);
        var colorClass = idx >= 0 ? 'companion-' + idx : 'companion-0';
        el.className = 'message ' + colorClass;
        el.style.borderLeftColor = getColor(idx >= 0 ? idx : 0);

        var html = '<div class="speaker" style="color:' + getColor(idx >= 0 ? idx : 0) + '">' + esc(data.speaker || speaker) + '</div>';
        html += '<div>' + esc(data.content || '') + '</div>';

        if (data.timestamp || data._ts) {
            var ts = data.timestamp || data._ts;
            try {
                var d = new Date(ts);
                html += '<div class="timestamp">' + d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) + '</div>';
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

        // Companion-select takes priority over button-filter when set
        var sel = document.getElementById('companion-select');
        var selectVal = sel ? sel.value : '';
        if (selectVal) {
            url += '&companion_id=' + encodeURIComponent(selectVal);
        } else if (activeFilters.size) {
            // Pass each selected companion as a separate param
            activeFilters.forEach(function (cid) {
                url += '&companion_id=' + encodeURIComponent(cid);
            });
        }

        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (msgs) {
                var lastDay = null;
                msgs.forEach(function (m) {
                    // Insert day markers
                    var day = m.timestamp ? m.timestamp.split('T')[0] : null;
                    if (day && day !== lastDay) {
                        var marker = document.createElement('div');
                        marker.className = 'day-marker';
                        marker.innerHTML = '<span>' + day + '</span>';
                        feed.appendChild(marker);
                        lastDay = day;
                    }
                    appendMessage({
                        speaker: m.speaker,
                        content: m.content,
                        timestamp: m.timestamp,
                    });
                });
                messageOffset += msgs.length;
                // Show Load More only if a full page was returned (more may exist)
                var loadMoreBtn = document.getElementById('btn-load-more');
                if (loadMoreBtn) {
                    loadMoreBtn.style.display = msgs.length >= LIMIT ? '' : 'none';
                }
                if (!append) autoScroll();
            })
            .catch(function () { /* silent */ });
    }

    // Wire up Load More button
    (function () {
        var btn = document.getElementById('btn-load-more');
        if (btn) {
            btn.addEventListener('click', function () {
                loadAllMessages(true);
            });
        }
    }());

    function autoScroll() {
        feed.scrollTop = feed.scrollHeight;
    }

    // -----------------------------------------------------------------------
    // State panel updater
    // -----------------------------------------------------------------------
    function updateStatePanel(companionId) {
        fetch('/api/observe/state?companion_id=' + companionId)
            .then(function (r) { return r.json(); })
            .then(function (s) {
                var prefix = companionId;
                var el;

                // Mood
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

                // Energy
                var energy = 50;
                if (s.energy && typeof s.energy === 'object') {
                    energy = s.energy.level != null ? s.energy.level : (s.energy.value != null ? s.energy.value : 50);
                } else if (typeof s.energy === 'number') {
                    energy = s.energy;
                }
                // Handle 0-1 scale
                if (energy <= 1) energy = Math.round(energy * 100);
                var energyPct = Math.max(0, Math.min(100, energy));
                el = document.getElementById(prefix + '-energy-label');
                if (el) el.textContent = energyPct + '%';
                var bar = document.getElementById(prefix + '-energy-bar');
                if (bar) {
                    bar.style.width = energyPct + '%';
                    bar.className = 'bar-fill energy' + (energyPct < 30 ? ' low' : energyPct < 60 ? ' medium' : '');
                }

                // Mode
                el = document.getElementById(prefix + '-mode');
                if (el) el.textContent = s.mode || '--';

                // Scene
                el = document.getElementById(prefix + '-scene');
                if (el) {
                    var sceneVal = '--';
                    if (s.scene && typeof s.scene === 'object') {
                        sceneVal = s.scene.activity || s.scene.location || JSON.stringify(s.scene);
                    } else if (s.scene) {
                        sceneVal = String(s.scene);
                    }
                    // Also check internal_state.was_doing
                    if (sceneVal === '--' && s.internal_state && s.internal_state.was_doing) {
                        sceneVal = s.internal_state.was_doing;
                    }
                    el.textContent = sceneVal;
                }

                // Closeness
                el = document.getElementById(prefix + '-closeness-label');
                if (el) el.textContent = s.closeness || 0;
                bar = document.getElementById(prefix + '-closeness-bar');
                if (bar) bar.style.width = Math.min(100, s.closeness || 0) + '%';

                // Emotion
                el = document.getElementById(prefix + '-emotion');
                if (el) el.textContent = s.emotion_profile || '--';

                // Romance
                el = document.getElementById(prefix + '-romance');
                if (el) el.textContent = (s.romance_level != null ? s.romance_level : '--');
            })
            .catch(function () { /* silent */ });
    }

    function pollStates() {
        companions.forEach(function (cid) {
            updateStatePanel(cid);
        });
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
    // Tab data
    // -----------------------------------------------------------------------
    function setTabCompanion(cid) {
        tabCompanion = cid;
        document.querySelectorAll('#tab-companion-toggle button').forEach(function (btn) {
            btn.className = btn.id === 'tab-cid-' + cid ? 'active-companion' : '';
        });
        pollActiveTab();
    }

    function pollActiveTab() {
        if (!tabCompanion) return;
        var cid = tabCompanion;
        switch (activeTab) {
            case 'facts': loadTabData('/api/observe/facts?companion_id=' + cid + '&limit=30', 'tab-facts', formatFacts); break;
            case 'opinions': loadTabData('/api/observe/opinions?companion_id=' + cid, 'tab-opinions', formatOpinions); break;
            case 'curiosity': loadTabData('/api/observe/curiosity?companion_id=' + cid, 'tab-curiosity', formatCuriosity); break;
            case 'goals': loadTabData('/api/observe/goals?companion_id=' + cid, 'tab-goals', formatGoals); break;
            case 'episodes': loadTabData('/api/observe/episodes?companion_id=' + cid + '&limit=10', 'tab-episodes', formatEpisodes); break;
            case 'relationship': loadTabData('/api/observe/relationship?companion_id=' + cid, 'tab-relationship', formatRelationship); break;
        }
    }

    function loadTabData(url, elId, formatter) {
        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                document.getElementById(elId).innerHTML = formatter(data);
            })
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
    // Tab switching
    // -----------------------------------------------------------------------
    document.querySelectorAll('.tab-bar button').forEach(function (btn) {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.tab-bar button').forEach(function (b) { b.classList.remove('active'); });
            document.querySelectorAll('.tab-content').forEach(function (c) { c.classList.remove('active'); });
            btn.classList.add('active');
            activeTab = btn.dataset.tab;
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

    function capitalize(s) {
        return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
    }

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
        var sliders = ['closeness-slider', 'romance-slider'];
        var inputs = ['emotion-input'];
        var editing = document.getElementById(cid + '-closeness-slider').style.display !== 'none';

        if (!editing) {
            // Enter edit mode — populate inputs with current values
            var closenessEl = document.getElementById(cid + '-closeness-label');
            var emotionEl = document.getElementById(cid + '-emotion');
            var romanceEl = document.getElementById(cid + '-romance');

            var closenessSlider = document.getElementById(cid + '-closeness-slider');
            closenessSlider.value = parseInt(closenessEl.textContent) || 0;
            closenessSlider.style.display = 'block';
            closenessSlider.oninput = function() { closenessEl.textContent = this.value; };

            var romanceSlider = document.getElementById(cid + '-romance-slider');
            romanceSlider.value = Math.round(parseFloat(romanceEl.textContent) || 0);
            romanceSlider.style.display = 'block';
            romanceSlider.oninput = function() { romanceEl.textContent = this.value; };

            var emotionInput = document.getElementById(cid + '-emotion-input');
            emotionInput.value = emotionEl.textContent !== '--' ? emotionEl.textContent : '';
            emotionInput.style.display = 'block';

            document.getElementById(cid + '-edit-btn').style.display = 'none';
            document.getElementById(cid + '-edit-actions').style.display = 'flex';
        }
    }

    window.cancelEdit = function(cid) {
        document.getElementById(cid + '-closeness-slider').style.display = 'none';
        document.getElementById(cid + '-romance-slider').style.display = 'none';
        document.getElementById(cid + '-emotion-input').style.display = 'none';
        document.getElementById(cid + '-edit-btn').style.display = '';
        document.getElementById(cid + '-edit-actions').style.display = 'none';
        // Re-poll to reset displayed values
        updateStatePanel(cid);
    };

    window.saveState = function(cid) {
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
            body: JSON.stringify(payload)
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.success) {
                // Exit edit mode
                window.cancelEdit(cid);
            } else {
                alert('Failed to save: ' + (data.error || 'unknown'));
            }
        })
        .catch(function(e) {
            alert('Save failed: ' + e.message);
        });
    };

    // Init on page load (also triggers on reconnect)
    if (typeof io === 'undefined') {
        // No socket.io yet, init manually after a beat
        setTimeout(init, 500);
    }

})();
