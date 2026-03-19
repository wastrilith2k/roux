/**
 * Companion Chat Frontend
 *
 * Connects to the Flask-SocketIO backend and provides a real-time chat
 * interface with typewriter effect on companion responses.
 */

(function () {
    'use strict';

    // ── DOM refs ──
    const $messages     = document.getElementById('messages');
    const $input        = document.getElementById('msg-input');
    const $sendBtn      = document.getElementById('send-btn');
    const $statusDot    = document.getElementById('status-dot');
    const $statusText   = document.getElementById('status-text');
    const $companionName = document.getElementById('companion-name');
    const $typing       = document.getElementById('typing-indicator');
    const $settingsOverlay = document.getElementById('settings-overlay');
    const $companionSelector = document.getElementById('companion-selector');

    // Settings inputs
    const $settingEmail     = document.getElementById('setting-email');
    const $settingCompanion = document.getElementById('setting-companion');
    const $settingSpeed     = document.getElementById('setting-speed');
    const $speedValue       = document.getElementById('speed-value');
    const $settingHistory   = document.getElementById('setting-history');
    const $historyValue     = document.getElementById('history-value');

    // ── Config (persisted to localStorage) ──
    const STORAGE_KEY = 'companion_chat_config';

    function loadConfig() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) return JSON.parse(raw);
        } catch (_) { /* ignore */ }
        return {};
    }

    function saveConfig(cfg) {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg)); } catch (_) { /* ignore */ }
    }

    let config = Object.assign({
        email: 'user@example.com',
        companionName: 'Companion',
        selectedCompanion: '',
        typewriterSpeed: 30,
        historyCount: 30,
    }, loadConfig());

    // Apply config to UI
    function applyConfig() {
        $settingEmail.value     = config.email;
        $settingCompanion.value = config.companionName;
        $settingSpeed.value     = config.typewriterSpeed;
        $speedValue.textContent = config.typewriterSpeed + 'ms per character';
        $settingHistory.value   = config.historyCount;
        $historyValue.textContent = config.historyCount + ' messages';
        $companionName.textContent = config.companionName;
    }

    function readConfigFromUI() {
        config.email           = $settingEmail.value.trim() || 'user@example.com';
        config.companionName   = $settingCompanion.value.trim() || 'Companion';
        config.typewriterSpeed = parseInt($settingSpeed.value, 10);
        config.historyCount    = parseInt($settingHistory.value, 10);
        $companionName.textContent = config.companionName;
        saveConfig(config);
    }

    // ── Settings panel ──
    function openSettings()  { $settingsOverlay.classList.add('open'); }
    function closeSettings() {
        readConfigFromUI();
        $settingsOverlay.classList.remove('open');
    }

    document.getElementById('settings-toggle').addEventListener('click', openSettings);
    document.getElementById('settings-close').addEventListener('click', closeSettings);
    document.getElementById('settings-backdrop').addEventListener('click', closeSettings);

    $settingSpeed.addEventListener('input', function () {
        $speedValue.textContent = this.value + 'ms per character';
    });
    $settingHistory.addEventListener('input', function () {
        $historyValue.textContent = this.value + ' messages';
    });

    document.getElementById('settings-reconnect').addEventListener('click', function () {
        readConfigFromUI();
        closeSettings();
        reconnect();
    });

    // ── Companion selector ──
    $companionSelector.addEventListener('change', function () {
        var selected = this.value;
        if (!selected || selected === config.selectedCompanion) return;
        config.selectedCompanion = selected;
        saveConfig(config);
        reconnect();
    });

    function fetchCompanions() {
        fetch('/api/observe/companions')
            .then(function (r) { return r.json(); })
            .then(function (companions) {
                if (!Array.isArray(companions) || companions.length === 0) return;
                $companionSelector.innerHTML = '';
                companions.forEach(function (cid) {
                    var opt = document.createElement('option');
                    opt.value = cid;
                    opt.textContent = cid.charAt(0).toUpperCase() + cid.slice(1);
                    $companionSelector.appendChild(opt);
                });
                // Select the stored companion, or default to first
                if (config.selectedCompanion && companions.indexOf(config.selectedCompanion) !== -1) {
                    $companionSelector.value = config.selectedCompanion;
                } else {
                    config.selectedCompanion = companions[0];
                    $companionSelector.value = companions[0];
                    saveConfig(config);
                }
            })
            .catch(function (err) {
                console.error('Failed to fetch companions:', err);
            });
    }

    // ── Socket.IO connection ──
    let socket = null;
    let waitingForResponse = false;

    // Active typewriter animations — track so we can skip/cancel
    let activeTypewriters = [];

    function setStatus(state, text) {
        $statusDot.className = 'status-dot ' + state;
        $statusText.textContent = text;
    }

    function connect() {
        if (socket) {
            socket.disconnect();
        }

        setStatus('connecting', 'Connecting...');

        var authPayload = { email: config.email };
        if (config.selectedCompanion) {
            authPayload.companion_id = config.selectedCompanion;
        }

        socket = io('/', {
            auth: authPayload,
            reconnection: true,
            reconnectionDelay: 2000,
            reconnectionAttempts: 20,
        });

        socket.on('connect', function () {
            setStatus('connected', 'Connected');
            // Request chat history
            socket.emit('request_history', { limit: config.historyCount });
        });

        socket.on('disconnect', function (reason) {
            setStatus('error', 'Disconnected');
            hideTyping();
        });

        socket.on('connect_error', function (err) {
            setStatus('error', 'Connection failed');
            console.error('Connection error:', err.message);
        });

        // ── Incoming events ──

        socket.on('status', function (data) {
            // Update companion name from server
            if (data.companion_name) {
                config.companionName = data.companion_name;
                $companionName.textContent = data.companion_name;
                $settingCompanion.value = data.companion_name;
                document.title = 'Chat with ' + data.companion_name;
            }
            if (data.companion_id) {
                config.selectedCompanion = data.companion_id;
                if ($companionSelector.querySelector('option[value="' + data.companion_id + '"]')) {
                    $companionSelector.value = data.companion_id;
                }
            }
            saveConfig(config);
        });

        socket.on('history', function (data) {
            renderHistory(data.messages || []);
        });

        socket.on('message', function (data) {
            hideTyping();
            waitingForResponse = false;
            addCompanionMessage(data);
        });

        socket.on('error', function (data) {
            hideTyping();
            waitingForResponse = false;
            addErrorMessage(data.message || 'Unknown error');
        });

        socket.on('message_queued', function (data) {
            addQueuedMessage(data.message || 'Message queued');
        });

        socket.on('command_response', function (data) {
            hideTyping();
            waitingForResponse = false;
            addCommandResponse(data);
        });

        socket.on('fact_approval_request', function (data) {
            // Show fact approval as a system message for now
            const text = 'Fact approval: "' + (data.fact || data.text || JSON.stringify(data)) + '"';
            addSystemMessage(text);
        });
    }

    function reconnect() {
        $messages.innerHTML = '';
        connect();
    }

    // ── Message rendering ──

    function scrollToBottom() {
        // Small delay to let DOM settle
        requestAnimationFrame(function () {
            $messages.scrollTop = $messages.scrollHeight;
        });
    }

    function isNearBottom() {
        var threshold = 80;
        return ($messages.scrollHeight - $messages.scrollTop - $messages.clientHeight) < threshold;
    }

    function formatTime(timestamp) {
        if (!timestamp) return '';
        try {
            var d = new Date(timestamp);
            if (isNaN(d.getTime())) return '';
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (_) {
            return '';
        }
    }

    function formatDate(timestamp) {
        if (!timestamp) return '';
        try {
            var d = new Date(timestamp);
            if (isNaN(d.getTime())) return '';
            var today = new Date();
            if (d.toDateString() === today.toDateString()) return 'Today';
            var yesterday = new Date(today);
            yesterday.setDate(today.getDate() - 1);
            if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
            return d.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
        } catch (_) {
            return '';
        }
    }

    function renderHistory(messages) {
        $messages.innerHTML = '';
        var lastDate = '';

        messages.forEach(function (msg) {
            // Day separator
            var dateStr = formatDate(msg.timestamp);
            if (dateStr && dateStr !== lastDate) {
                lastDate = dateStr;
                var sep = document.createElement('div');
                sep.className = 'day-separator';
                sep.innerHTML = '<span>' + escapeHtml(dateStr) + '</span>';
                $messages.appendChild(sep);
            }

            if (msg.role === 'user') {
                addUserMessageEl(msg.content, msg.timestamp);
            } else {
                addCompanionMessageEl(msg.content, null, msg.timestamp, null, false);
            }
        });

        scrollToBottom();
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function addUserMessageEl(text, timestamp) {
        var el = document.createElement('div');
        el.className = 'msg msg-user';

        var timeStr = formatTime(timestamp);
        if (timeStr) {
            el.innerHTML = '<div class="msg-text">' + escapeHtml(text) + '</div>' +
                           '<div class="msg-time" style="text-align:right;">' + escapeHtml(timeStr) + '</div>';
        } else {
            el.innerHTML = '<div class="msg-text">' + escapeHtml(text) + '</div>';
        }

        $messages.appendChild(el);
        return el;
    }

    // addCompanionMessageEl is defined later with mood emoji support

    function appendModelBadge(el) {
        if (el.dataset.model) {
            var badge = document.createElement('div');
            badge.className = 'msg-model';
            badge.textContent = el.dataset.model;
            el.appendChild(badge);
        }
    }

    // ── Typewriter effect ──
    function runTypewriter(span, text, msgEl) {
        var i = 0;
        var speed = config.typewriterSpeed;
        var wasNearBottom = isNearBottom();

        // Cursor element
        var cursor = document.createElement('span');
        cursor.className = 'typewriter-cursor';
        cursor.textContent = '\u258F'; // thin block cursor
        span.after(cursor);

        var entry = { cancelled: false };
        activeTypewriters.push(entry);

        function tick() {
            if (entry.cancelled) {
                // Dump remaining text
                span.textContent = text;
                cursor.remove();
                appendModelBadge(msgEl);
                if (wasNearBottom) scrollToBottom();
                return;
            }
            if (i < text.length) {
                span.textContent += text[i];
                i++;
                if (wasNearBottom) scrollToBottom();
                setTimeout(tick, speed);
            } else {
                // Done
                cursor.remove();
                appendModelBadge(msgEl);
                // Remove from active list
                var idx = activeTypewriters.indexOf(entry);
                if (idx !== -1) activeTypewriters.splice(idx, 1);
                if (wasNearBottom) scrollToBottom();
            }
        }

        tick();
    }

    function addCompanionMessage(data) {
        var text = data.text || data.content || data.message_text || '';
        var sender = data.sender_name || null;
        var model = data.model_used || null;
        var ts = data.timestamp || new Date().toISOString();
        addCompanionMessageEl(text, sender, ts, model, true);
    }

    function addUserMessage(text) {
        addUserMessageEl(text, new Date().toISOString());
        scrollToBottom();
    }

    function addErrorMessage(text) {
        var el = document.createElement('div');
        el.className = 'msg msg-error';
        el.textContent = text;
        $messages.appendChild(el);
        scrollToBottom();
    }

    function addSystemMessage(text) {
        var el = document.createElement('div');
        el.className = 'msg msg-system';
        el.textContent = text;
        $messages.appendChild(el);
        scrollToBottom();
    }

    function addQueuedMessage(text) {
        var el = document.createElement('div');
        el.className = 'msg msg-queued';
        el.textContent = text;
        $messages.appendChild(el);
        scrollToBottom();
    }

    function addCommandResponse(data) {
        var el = document.createElement('div');
        el.className = 'msg msg-command';

        var cmd = data.command || '/command';
        var text = data.text || '';
        if (data.error) {
            text = 'Error: ' + data.error;
        }

        el.innerHTML = '<div class="msg-meta">' +
                        '<span class="msg-sender">' + escapeHtml(cmd) + '</span>' +
                        '</div>' +
                        '<div class="msg-text">' + escapeHtml(text) + '</div>';

        $messages.appendChild(el);
        scrollToBottom();
    }

    // ── Typing indicator ──
    function showTyping() {
        $typing.classList.add('visible');
        scrollToBottom();
    }

    function hideTyping() {
        $typing.classList.remove('visible');
    }

    // ── Send message ──
    function sendMessage() {
        var text = $input.value.trim();
        if (!text) return;
        if (!socket || !socket.connected) {
            addErrorMessage('Not connected. Check settings and reconnect.');
            return;
        }

        $input.value = '';
        autoResizeInput();

        // Show user message immediately
        addUserMessage(text);

        // Show typing indicator
        showTyping();
        waitingForResponse = true;

        // Send to backend
        socket.emit('send_message', { message: text });
    }

    // ── Input handling ──
    $sendBtn.addEventListener('click', sendMessage);

    $input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    function autoResizeInput() {
        $input.style.height = 'auto';
        $input.style.height = Math.min($input.scrollHeight, 140) + 'px';
    }

    $input.addEventListener('input', autoResizeInput);

    // Click on a companion message to skip its typewriter animation
    $messages.addEventListener('click', function (e) {
        // Skip all active typewriters on any click in messages area
        if (activeTypewriters.length > 0) {
            activeTypewriters.forEach(function (entry) { entry.cancelled = true; });
            activeTypewriters = [];
        }
    });

    // ── Mood emoji mapping ──
    var MOOD_EMOJI = {
        'neutral': '',
        'relaxed': '\uD83D\uDE0C',     // relieved face
        'focused': '\uD83E\uDDD0',     // thinking face
        'playful': '\uD83D\uDE1C',     // winking tongue
        'content': '\uD83D\uDE0A',     // smiling
        'restless': '\uD83D\uDE15',    // confused
        'thoughtful': '\uD83E\uDD14',  // thinking
        'tired': '\uD83D\uDE34',       // sleeping
        'energetic': '\u26A1',         // lightning
        'nostalgic': '\uD83E\uDD79',   // face holding tears
        'creative': '\uD83C\uDFA8',    // palette
        'irritated': '\uD83D\uDE24',   // persevering
        'anxious': '\uD83D\uDE30',     // anxious
        'melancholy': '\uD83D\uDE14',  // pensive
        'giddy': '\uD83E\uDD29',      // star struck
        'suspicious': '\uD83E\uDD28',  // face with raised eyebrow
        'protective': '\uD83D\uDEE1\uFE0F', // shield
    };

    var currentMood = '';

    function getMoodEmoji(mood) {
        if (!mood) return '';
        return MOOD_EMOJI[mood.toLowerCase()] || '\uD83D\uDE10'; // neutral face fallback
    }

    // ── Companion info polling ──
    function pollCompanionInfo() {
        if (!socket || !socket.connected) return;

        // Request state via socketio
        socket.emit('get_state');

        // Also try REST for schedule
        fetch('/api/observe/state?companion_id=' + encodeURIComponent(config.selectedCompanion || config.companionName.toLowerCase()))
            .then(function (r) { return r.json(); })
            .then(function (s) {
                // Mood
                var mood = '--';
                if (s.internal_state && s.internal_state.mood) {
                    mood = s.internal_state.mood;
                } else if (s.mood && typeof s.mood === 'string') {
                    mood = s.mood;
                }
                currentMood = mood;
                var emoji = getMoodEmoji(mood);
                document.getElementById('mood-value').innerHTML = emoji ? emoji + ' ' + escapeHtml(mood) : escapeHtml(mood);

                // Energy
                var energy = '--';
                if (s.internal_state && s.internal_state.energy != null) {
                    energy = Math.round(s.internal_state.energy * 100) + '%';
                } else if (typeof s.energy === 'number') {
                    var e = s.energy <= 1 ? Math.round(s.energy * 100) : Math.round(s.energy);
                    energy = e + '%';
                }
                document.getElementById('energy-value').textContent = energy;

                // Scene / activity
                var scene = '--';
                if (s.internal_state && s.internal_state.was_doing) {
                    scene = s.internal_state.was_doing;
                } else if (s.scene && typeof s.scene === 'string') {
                    scene = s.scene;
                } else if (s.scene && s.scene.activity) {
                    scene = s.scene.activity;
                }
                document.getElementById('scene-value').textContent = scene;
            })
            .catch(function () { /* silent */ });

        // Schedule
        fetch('/api/observe/state?companion_id=' + encodeURIComponent(config.selectedCompanion || config.companionName.toLowerCase()))
            .then(function (r) { return r.json(); })
            .then(function (s) {
                // Try to get schedule from internal state or a dedicated endpoint
                var schedule = '--';
                if (s.internal_state && s.internal_state.life_event) {
                    schedule = s.internal_state.life_event;
                }
                document.getElementById('schedule-value').textContent = schedule;
            })
            .catch(function () { /* silent */ });
    }

    // ── Companion message with mood emoji ──
    function addCompanionMessageEl(text, senderName, timestamp, modelUsed, typewriter) {
        var el = document.createElement('div');
        el.className = 'msg msg-companion';

        var name = senderName || config.companionName;
        var timeStr = formatTime(timestamp);
        var emoji = getMoodEmoji(currentMood);

        // Meta line with mood emoji
        var metaHtml = '<div class="msg-meta">' +
                       (emoji ? '<span class="msg-mood">' + emoji + '</span>' : '') +
                       '<span class="msg-sender">' + escapeHtml(name) + '</span>' +
                       (timeStr ? '<span class="msg-time">' + escapeHtml(timeStr) + '</span>' : '') +
                       '</div>';

        var textSpan = document.createElement('span');
        textSpan.className = 'msg-text';

        el.innerHTML = metaHtml;
        el.appendChild(textSpan);

        if (modelUsed) {
            el.dataset.model = modelUsed;
        }

        $messages.appendChild(el);

        if (typewriter && text) {
            runTypewriter(textSpan, text, el);
        } else {
            textSpan.textContent = text || '';
            appendModelBadge(el);
            scrollToBottom();
        }

        return el;
    };

    // ── Init ──
    applyConfig();
    fetchCompanions();
    connect();
    $input.focus();

    // Poll companion info every 10s
    setInterval(pollCompanionInfo, 10000);
    setTimeout(pollCompanionInfo, 2000); // Initial poll after connect

})();
