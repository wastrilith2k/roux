/* Observation Dashboard — Vanilla JS */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let activeTab = 'facts';
    let tabCompanion = 'kai';
    let connected = false;
    let statePollerHandle = null;
    let dataPollerHandle = null;

    const feed = document.getElementById('feed');

    // -----------------------------------------------------------------------
    // SocketIO connection
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
        // Load initial messages
        loadMessages('kai');
        loadMessages('mira');
    });

    socket.on('disconnect', function () {
        connected = false;
        document.getElementById('status-dot').className = 'status-dot';
        document.getElementById('status-text').textContent = 'Disconnected';
    });

    // -----------------------------------------------------------------------
    // Simulation events
    // -----------------------------------------------------------------------
    socket.on('sim:message', function (data) {
        appendMessage(data);
        autoScroll();
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
    // Message rendering
    // -----------------------------------------------------------------------
    function appendMessage(data) {
        var el = document.createElement('div');
        var speaker = (data.speaker || '').toLowerCase();
        el.className = 'message ' + (speaker === 'mira' ? 'mira' : 'kai');

        var html = '<div class="speaker">' + esc(data.speaker || speaker) + '</div>';
        html += '<div>' + esc(data.content || '') + '</div>';

        if (data.timestamp || data._ts) {
            var ts = data.timestamp || data._ts;
            try {
                var d = new Date(ts);
                html += '<div class="timestamp">' + d.toLocaleTimeString() + '</div>';
            } catch (e) { /* ignore */ }
        }

        if (data.analysis) {
            html += '<details><summary>analysis</summary><div class="analysis-content">';
            var a = data.analysis;
            if (typeof a === 'string') {
                html += esc(a);
            } else {
                for (var k in a) {
                    html += '<strong>' + esc(k) + ':</strong> ' + esc(String(a[k])) + '<br>';
                }
            }
            html += '</div></details>';
        }

        el.innerHTML = html;
        feed.appendChild(el);
    }

    function loadMessages(companionId) {
        fetch('/api/observe/messages?companion_id=' + companionId + '&limit=30')
            .then(function (r) { return r.json(); })
            .then(function (msgs) {
                msgs.forEach(function (m) {
                    appendMessage({
                        speaker: m.speaker || companionId,
                        content: m.content,
                        timestamp: m.timestamp,
                    });
                });
                autoScroll();
            })
            .catch(function () { /* silent */ });
    }

    function autoScroll() {
        feed.scrollTop = feed.scrollHeight;
    }

    // -----------------------------------------------------------------------
    // State panel updater (polls every 5s)
    // -----------------------------------------------------------------------
    function updateStatePanel(companionId) {
        fetch('/api/observe/state?companion_id=' + companionId)
            .then(function (r) { return r.json(); })
            .then(function (s) {
                var prefix = companionId;

                // Mood
                var moodEl = document.getElementById(prefix + '-mood');
                var moodVal = '--';
                if (s.mood && typeof s.mood === 'object') {
                    moodVal = s.mood.primary || s.mood.label || JSON.stringify(s.mood);
                } else if (s.mood) {
                    moodVal = String(s.mood);
                }
                moodEl.textContent = moodVal;

                // Energy
                var energy = 50;
                if (s.energy && typeof s.energy === 'object') {
                    energy = s.energy.level != null ? s.energy.level : (s.energy.value != null ? s.energy.value : 50);
                } else if (typeof s.energy === 'number') {
                    energy = s.energy;
                }
                var energyPct = Math.max(0, Math.min(100, energy));
                document.getElementById(prefix + '-energy-label').textContent = energyPct + '%';
                var bar = document.getElementById(prefix + '-energy-bar');
                bar.style.width = energyPct + '%';
                bar.className = 'bar-fill energy' + (energyPct < 30 ? ' low' : energyPct < 60 ? ' medium' : '');

                // Mode
                document.getElementById(prefix + '-mode').textContent = s.mode || '--';

                // Scene
                var sceneVal = '--';
                if (s.scene && typeof s.scene === 'object') {
                    sceneVal = s.scene.activity || s.scene.location || JSON.stringify(s.scene);
                } else if (s.scene) {
                    sceneVal = String(s.scene);
                }
                document.getElementById(prefix + '-scene').textContent = sceneVal;

                // Closeness
                var closeness = s.closeness || 0;
                document.getElementById(prefix + '-closeness-label').textContent = closeness;
                document.getElementById(prefix + '-closeness-bar').style.width = Math.min(100, closeness) + '%';

                // Emotion profile
                document.getElementById(prefix + '-emotion').textContent = s.emotion_profile || '--';
            })
            .catch(function () { /* silent */ });
    }

    function pollStates() {
        updateStatePanel('kai');
        updateStatePanel('mira');
    }

    // -----------------------------------------------------------------------
    // Simulation status (polls every 5s alongside state)
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
    // Tab data (polls every 15s, only active tab)
    // -----------------------------------------------------------------------
    function pollActiveTab() {
        var cid = tabCompanion;
        switch (activeTab) {
            case 'facts': loadFacts(cid); break;
            case 'opinions': loadOpinions(cid); break;
            case 'curiosity': loadCuriosity(cid); break;
            case 'goals': loadGoals(cid); break;
            case 'episodes': loadEpisodes(cid); break;
            case 'relationship': loadRelationship(cid); break;
        }
    }

    function loadFacts(cid) {
        fetch('/api/observe/facts?companion_id=' + cid + '&limit=30')
            .then(function (r) { return r.json(); })
            .then(function (facts) {
                var el = document.getElementById('tab-facts');
                if (!facts.length) { el.innerHTML = '<div class="empty-state">No facts yet</div>'; return; }
                var html = '<table class="data-table"><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Confidence</th><th>Updated</th></tr>';
                facts.forEach(function (f) {
                    html += '<tr><td>' + esc(f.subject) + '</td><td>' + esc(f.predicate) +
                        '</td><td>' + esc(f.object) + '</td><td>' + confidenceBadge(f.confidence) +
                        '</td><td>' + shortDate(f.updated_at) + '</td></tr>';
                });
                html += '</table>';
                el.innerHTML = html;
            })
            .catch(function () {});
    }

    function loadOpinions(cid) {
        fetch('/api/observe/opinions?companion_id=' + cid)
            .then(function (r) { return r.json(); })
            .then(function (opinions) {
                var el = document.getElementById('tab-opinions');
                if (!opinions.length) { el.innerHTML = '<div class="empty-state">No opinions yet</div>'; return; }
                var html = '<table class="data-table"><tr><th>Topic</th><th>Opinion</th><th>Confidence</th><th>Evidence</th></tr>';
                opinions.forEach(function (o) {
                    html += '<tr><td>' + esc(o.topic) + '</td><td>' + esc(truncate(o.opinion, 120)) +
                        '</td><td>' + confidenceBadge(o.confidence) +
                        '</td><td>' + (o.evidence_count || 0) + '</td></tr>';
                });
                html += '</table>';
                el.innerHTML = html;
            })
            .catch(function () {});
    }

    function loadCuriosity(cid) {
        fetch('/api/observe/curiosity?companion_id=' + cid)
            .then(function (r) { return r.json(); })
            .then(function (threads) {
                var el = document.getElementById('tab-curiosity');
                if (!threads.length) { el.innerHTML = '<div class="empty-state">No curiosity threads yet</div>'; return; }
                var html = '<table class="data-table"><tr><th>Topic</th><th>Category</th><th>Urgency</th><th>Times Asked</th></tr>';
                threads.forEach(function (t) {
                    html += '<tr><td>' + esc(t.topic || '') + '</td><td>' + esc(t.category || '') +
                        '</td><td>' + confidenceBadge(t.urgency) +
                        '</td><td>' + (t.times_asked || 0) + '</td></tr>';
                });
                html += '</table>';
                el.innerHTML = html;
            })
            .catch(function () {});
    }

    function loadGoals(cid) {
        fetch('/api/observe/goals?companion_id=' + cid)
            .then(function (r) { return r.json(); })
            .then(function (goals) {
                var el = document.getElementById('tab-goals');
                if (!goals.length) { el.innerHTML = '<div class="empty-state">No goals yet</div>'; return; }
                var html = '<table class="data-table"><tr><th>Goal</th><th>Category</th><th>Progress</th><th>Motivation</th></tr>';
                goals.forEach(function (g) {
                    var pct = Math.round((g.progress || 0) * 100);
                    html += '<tr><td>' + esc(g.goal) + '</td><td>' + esc(g.category || '') +
                        '</td><td>' + pct + '%</td><td>' + esc(truncate(g.motivation || '', 80)) + '</td></tr>';
                });
                html += '</table>';
                el.innerHTML = html;
            })
            .catch(function () {});
    }

    function loadEpisodes(cid) {
        fetch('/api/observe/episodes?companion_id=' + cid + '&limit=10')
            .then(function (r) { return r.json(); })
            .then(function (episodes) {
                var el = document.getElementById('tab-episodes');
                if (!episodes.length) { el.innerHTML = '<div class="empty-state">No episodes yet</div>'; return; }
                var html = '<table class="data-table"><tr><th>When</th><th>Summary</th><th>Significance</th><th>Emotional Arc</th></tr>';
                episodes.forEach(function (e) {
                    html += '<tr><td>' + shortDate(e.started_at) + '</td><td>' + esc(truncate(e.summary || '', 150)) +
                        '</td><td>' + (e.significance || '--') +
                        '</td><td>' + esc(e.emotional_arc || '--') + '</td></tr>';
                });
                html += '</table>';
                el.innerHTML = html;
            })
            .catch(function () {});
    }

    function loadRelationship(cid) {
        fetch('/api/observe/relationship?companion_id=' + cid)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var el = document.getElementById('tab-relationship');
                if (!data.evaluation) { el.innerHTML = '<div class="empty-state">No relationship data yet</div>'; return; }
                var eval_ = data.evaluation;
                var html = '<div style="padding: 8px;">';
                if (typeof eval_ === 'object') {
                    for (var key in eval_) {
                        html += '<div style="margin-bottom: 8px;"><strong>' + esc(key) + ':</strong> ';
                        var v = eval_[key];
                        html += typeof v === 'object' ? '<pre style="margin:4px 0;color:#8b949e;">' + esc(JSON.stringify(v, null, 2)) + '</pre>' : esc(String(v));
                        html += '</div>';
                    }
                } else {
                    html += esc(String(eval_));
                }
                if (data.evaluated_at) {
                    html += '<div style="margin-top:8px;color:#484f58;font-size:11px;">Last evaluated: ' + shortDate(data.evaluated_at) + '</div>';
                }
                html += '</div>';
                el.innerHTML = html;
            })
            .catch(function () {});
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

    // Companion toggle in tabs (exposed globally for onclick)
    window.setTabCompanion = function (cid) {
        tabCompanion = cid;
        document.getElementById('tab-cid-kai').className = cid === 'kai' ? 'active-kai' : '';
        document.getElementById('tab-cid-mira').className = cid === 'mira' ? 'active-mira' : '';
        pollActiveTab();
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    function esc(s) {
        if (!s) return '';
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
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
    // Start polling
    // -----------------------------------------------------------------------
    pollStates();
    pollSimStatus();
    pollActiveTab();

    statePollerHandle = setInterval(function () {
        pollStates();
        pollSimStatus();
    }, 5000);

    dataPollerHandle = setInterval(pollActiveTab, 15000);

})();
