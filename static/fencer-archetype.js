/**
 * Fencer style archetypes + targeting signatures from bout touch predictions.
 * MVP: model attack label "other" is treated as Counter.
 */
(function (global) {
    'use strict';

    var ACTION_KEYS = ['lunge', 'counter', 'fleche', 'other'];
    var TARGET_KEYS = ['chest', 'abdomen', 'arm', 'leg'];

    var ARCHETYPE_META = {
        duelist: {
            id: 'duelist',
            name: 'The Duelist',
            style: 'Attack-focused fencer who creates opportunities through direct attacks.',
        },
        sniper: {
            id: 'sniper',
            name: 'The Sniper',
            style: 'Precision fencer who waits for openings and scores through counters.',
        },
        bolt: {
            id: 'bolt',
            name: 'The Bolt',
            style: 'Explosive fencer who relies on fast, decisive attacks.',
        },
        tactician: {
            id: 'tactician',
            name: 'The Tactician',
            style: 'Balanced fencer who scores through a varied mix of actions.',
        },
        hybrid: {
            id: 'hybrid',
            name: 'The Hybrid',
            style: 'A fencer who combines two major scoring approaches rather than relying on one.',
        },
    };

    var TARGETING_META = {
        arm_hunter: { id: 'arm_hunter', name: 'Arm Hunter' },
        body_hunter: { id: 'body_hunter', name: 'Body Hunter' },
        leg_hunter: { id: 'leg_hunter', name: 'Leg Hunter' },
        standard: { id: 'standard', name: 'Standard Target Profile' },
    };

    function roundPct(n, total) {
        if (!total) return 0;
        return Math.round((n / total) * 100);
    }

    function normalizeAttackLabel(label) {
        if (!label) return null;
        var s = String(label).toLowerCase();
        if (s === 'other') return 'counter';
        if (s === 'counter' || s === 'lunge' || s === 'fleche') return s;
        return null;
    }

    function attackActionPcts(touches) {
        var counts = { lunge: 0, counter: 0, fleche: 0, other: 0 };
        var classified = 0;
        (touches || []).forEach(function (t) {
            var mapped = normalizeAttackLabel(t.attack_prediction);
            if (!mapped) return;
            classified += 1;
            counts[mapped] += 1;
        });
        var pcts = {
            lunge: roundPct(counts.lunge, classified),
            counter: roundPct(counts.counter, classified),
            fleche: roundPct(counts.fleche, classified),
            other: roundPct(counts.other, classified),
        };
        return { counts: counts, pcts: pcts, classifiedCount: classified };
    }

    function targetZonePcts(touches) {
        var counts = { chest: 0, abdomen: 0, arm: 0, leg: 0 };
        var classified = 0;
        (touches || []).forEach(function (t) {
            var p = t.prediction && String(t.prediction).toLowerCase();
            if (!p || !Object.prototype.hasOwnProperty.call(counts, p)) return;
            counts[p] += 1;
            classified += 1;
        });
        var pcts = {
            chest: roundPct(counts.chest, classified),
            abdomen: roundPct(counts.abdomen, classified),
            arm: roundPct(counts.arm, classified),
            leg: roundPct(counts.leg, classified),
        };
        return { counts: counts, pcts: pcts, classifiedCount: classified };
    }

    function topTwoActions(pcts) {
        var ranked = ACTION_KEYS.map(function (k) {
            return { key: k, pct: pcts[k] || 0 };
        }).sort(function (a, b) {
            return b.pct - a.pct;
        });
        return ranked.slice(0, 2);
    }

    function anyActionExceeds(pcts, limit) {
        return ACTION_KEYS.some(function (k) {
            return (pcts[k] || 0) > limit;
        });
    }

    function fallbackFromTopAction(pcts) {
        var ranked = [
            { key: 'lunge', id: 'duelist' },
            { key: 'counter', id: 'sniper' },
            { key: 'fleche', id: 'bolt' },
        ].sort(function (a, b) {
            return (pcts[b.key] || 0) - (pcts[a.key] || 0);
        });
        if ((pcts[ranked[0].key] || 0) > 0) {
            return ARCHETYPE_META[ranked[0].id];
        }
        return ARCHETYPE_META.hybrid;
    }

    function classifyArchetypeComparative(userAction, oppAction, context) {
        var pcts = (userAction && userAction.pcts) || {};
        var oppPcts = (oppAction && oppAction.pcts) || {};
        var ctx = context || {};
        var userArm = Number(ctx.userArm) || 0;
        var oppArm = Number(ctx.oppArm) || 0;
        var userPre = Number(ctx.userPre) || 0;
        var oppPre = Number(ctx.oppPre) || 0;
        var advancesMore = userPre > oppPre;
        var retreatsMore = oppPre > userPre;
        var pressesBlade = userArm > oppArm;

        var lunge = pcts.lunge || 0;
        var counter = pcts.counter || 0;
        var fleche = pcts.fleche || 0;
        var other = pcts.other || 0;
        var oppLunge = oppPcts.lunge || 0;
        var oppCounter = oppPcts.counter || 0;

        if (lunge >= 57) return ARCHETYPE_META.duelist;
        if (counter >= 40) return ARCHETYPE_META.sniper;
        if (fleche >= 23) return ARCHETYPE_META.bolt;

        if (lunge >= 45 && advancesMore && (pressesBlade || lunge >= oppLunge)) {
            return ARCHETYPE_META.duelist;
        }
        if (counter >= 30 && (retreatsMore || counter > oppCounter)) {
            return ARCHETYPE_META.sniper;
        }
        if (fleche >= 15 && fleche >= lunge && fleche >= counter) {
            return ARCHETYPE_META.bolt;
        }
        if (lunge >= 40 && advancesMore && pressesBlade) {
            return ARCHETYPE_META.duelist;
        }

        var tactician =
            lunge >= 20 && lunge <= 55 &&
            counter >= 20 && counter <= 50 &&
            fleche >= 10 && fleche <= 35 &&
            other < 20 &&
            !anyActionExceeds(pcts, 55);
        if (tactician) return ARCHETYPE_META.tactician;

        var top = topTwoActions(pcts);
        if (
            top.length === 2 &&
            Math.abs(top[0].pct - top[1].pct) <= 10 &&
            top[0].pct + top[1].pct >= 65 &&
            top[0].key !== 'other'
        ) {
            return ARCHETYPE_META.hybrid;
        }

        return fallbackFromTopAction(pcts);
    }

    function classifyArchetype(actionInfo) {
        return classifyArchetypeComparative(actionInfo, null, null);
    }

    function classifyTargeting(targetInfo) {
        var pcts = targetInfo.pcts;
        if (targetInfo.classifiedCount <= 0) return TARGETING_META.standard;
        if ((pcts.arm || 0) >= 28) return TARGETING_META.arm_hunter;
        if ((pcts.abdomen || 0) >= 18) return TARGETING_META.body_hunter;
        if ((pcts.leg || 0) >= 13) return TARGETING_META.leg_hunter;
        return TARGETING_META.standard;
    }

    function boutContextFromCaches(userFencer) {
        var arm = (typeof global.ARM_ATTEMPTS_CACHE !== 'undefined' && global.ARM_ATTEMPTS_CACHE)
            ? global.ARM_ATTEMPTS_CACHE
            : (typeof ARM_ATTEMPTS_CACHE !== 'undefined' ? ARM_ATTEMPTS_CACHE : null);
        var pre = (arm && arm.pre_touch_aggressor) ? arm.pre_touch_aggressor : {};
        var is1 = userFencer !== 'fencer2';
        return {
            userArm: is1 ? (arm && arm.fencer1_total) || 0 : (arm && arm.fencer2_total) || 0,
            oppArm: is1 ? (arm && arm.fencer2_total) || 0 : (arm && arm.fencer1_total) || 0,
            userPre: is1 ? (pre.fencer1_pre_touch_aggression || 0) : (pre.fencer2_pre_touch_aggression || 0),
            oppPre: is1 ? (pre.fencer2_pre_touch_aggression || 0) : (pre.fencer1_pre_touch_aggression || 0),
        };
    }

    function classifyUserProfile(userTouches, oppTouches, userFencer) {
        var userAction = attackActionPcts(userTouches);
        var oppAction = attackActionPcts(oppTouches);
        var targetInfo = targetZonePcts(userTouches);
        var ctx = boutContextFromCaches(userFencer);
        var archetype = classifyArchetypeComparative(userAction, oppAction, ctx);
        var targeting = classifyTargeting(targetInfo);
        return {
            archetype: archetype,
            targeting: targeting,
            styleBlurb: archetype.style,
            // Rationale comes from the LLM once analysis finishes — not canned text.
            rationale: '',
            targetingNotes: '',
            actionPcts: userAction.pcts,
            classifiedCount: userAction.classifiedCount,
            fromRules: true,
        };
    }

    function classifyFencerProfile(touches) {
        var actionInfo = attackActionPcts(touches);
        var targetInfo = targetZonePcts(touches);
        var archetype = classifyArchetype(actionInfo);
        var targeting = classifyTargeting(targetInfo);
        return {
            archetype: archetype,
            targeting: targeting,
            styleBlurb: archetype.style,
            actionPcts: actionInfo.pcts,
            actionCounts: actionInfo.counts,
            targetPcts: targetInfo.pcts,
            targetCounts: targetInfo.counts,
            classifiedCount: actionInfo.classifiedCount,
            targetClassifiedCount: targetInfo.classifiedCount,
        };
    }

    function displayAttackLabel(label) {
        var mapped = normalizeAttackLabel(label);
        if (mapped === 'counter') return 'Counter / Unconventional Strike';
        if (mapped === 'lunge') return 'Lunge';
        if (mapped === 'fleche') return 'Fleche';
        if (!label) return '';
        return String(label).charAt(0).toUpperCase() + String(label).slice(1);
    }

    function shouldShowReveal(opts) {
        opts = opts || {};
        if (global.CLIPPER_MODE) return false;
        if (document.body.classList.contains('touch-focus-mode')) return false;
        if (opts.touchFocus) return false;
        if (typeof TouchClip !== 'undefined' && TouchClip.readTouchParamFromUrl) {
            try {
                if (TouchClip.readTouchParamFromUrl()) return false;
            } catch (e) { /* ignore */ }
        }
        if (global._pendingTouchFocusId) return false;
        var f1 = opts.fencer1Touches || [];
        var f2 = opts.fencer2Touches || [];
        var hasAttack = f1.concat(f2).some(function (t) {
            return !!t.attack_prediction;
        });
        return hasAttack;
    }

    function ensureStyles() {
        var old = document.getElementById('fencer-archetype-styles');
        if (old) old.remove();
        var old2 = document.getElementById('fencer-archetype-styles-v2');
        if (old2) old2.remove();
        var old3 = document.getElementById('fencer-archetype-styles-v3');
        if (old3) old3.remove();
        var old4 = document.getElementById('fencer-archetype-styles-v4');
        if (old4) old4.remove();
        var old5 = document.getElementById('fencer-archetype-styles-v5');
        if (old5) old5.remove();
        if (document.getElementById('fencer-archetype-styles-v6')) return;
        var link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Syne:wght@600;700;800&display=swap';
        document.head.appendChild(link);

        var style = document.createElement('style');
        style.id = 'fencer-archetype-styles-v6';
        style.textContent = [
            '.archetype-reveal {',
            '  position: relative;',
            '  border-radius: 1.35rem;',
            '  overflow: hidden;',
            '  border: 1px solid rgba(232,236,240,0.1);',
            '  font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif;',
            '  background:',
            '    radial-gradient(ellipse 80% 55% at 10% 0%, rgba(20,184,166,0.16), transparent 55%),',
            '    radial-gradient(ellipse 50% 40% at 100% 0%, rgba(200,245,66,0.06), transparent 50%),',
            '    linear-gradient(165deg, #0c1016 0%, #12171f 100%);',
            '  box-shadow: 0 18px 48px rgba(0,0,0,0.4);',
            '}',
            '.archetype-reveal .arch-video-shell {',
            '  position: relative;',
            '  border-radius: 0.85rem;',
            '  overflow: hidden;',
            '  background: #000;',
            '  border: 1px solid rgba(148,163,184,0.14);',
            '  line-height: 0;',
            '}',
            '.archetype-reveal .arch-video-shell video {',
            '  display: block; width: 100%; height: auto;',
            '  max-height: min(34vh, 320px); object-fit: contain; background: #000;',
            '}',
            '.archetype-reveal .arch-title,',
            '.archetype-reveal .arch-hero-title,',
            '.archetype-reveal .arch-practice-heading,',
            '#llmCoachingPanel .llm-coach-heading {',
            '  font-family: Syne, "DM Sans", ui-sans-serif, system-ui, sans-serif;',
            '  font-weight: 700;',
            '  letter-spacing: -0.02em;',
            '  line-height: 1.25;',
            '  overflow: visible;',
            '}',
            '.archetype-reveal .arch-card {',
            '  border-radius: 1rem;',
            '  padding: 1.15rem 1.2rem;',
            '  background: rgba(7,9,12,0.72);',
            '  border: 1px solid rgba(232,236,240,0.1);',
            '  border-left: 3px solid #2dd4bf;',
            '  opacity: 0;',
            '  transform: translateY(12px);',
            '  animation: archCardIn 0.5s ease forwards;',
            '}',
            '.archetype-reveal .arch-card.arch-practice-section { border-left-color: #c8f542; animation-delay: 0.12s; }',
            '.archetype-reveal .arch-drill-grid {',
            '  display: grid;',
            '  grid-template-columns: 1fr;',
            '  gap: 0.55rem;',
            '}',
            '@media (min-width: 640px) {',
            '  .archetype-reveal .arch-drill-grid { grid-template-columns: 1fr 1fr; }',
            '}',
            '.archetype-reveal .arch-eyebrow {',
            '  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700;',
            '  color: #5eead4;',
            '}',
            '.archetype-reveal .arch-badge {',
            '  display: inline-flex; align-items: center;',
            '  font-size: 0.72rem; font-weight: 600;',
            '  padding: 0.3rem 0.65rem; border-radius: 999px;',
            '  background: rgba(20,184,166,0.12); color: #99f6e4;',
            '  border: 1px solid rgba(45,212,191,0.28);',
            '}',
            '.archetype-reveal .arch-hero-label {',
            '  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;',
            '  color: #94a3b8; margin: 0;',
            '}',
            '.archetype-reveal .arch-hero-title {',
            '  font-size: 1.35rem; color: #f8fafc; margin: 0.25rem 0 0.75rem;',
            '}',
            '.archetype-reveal .arch-cta {',
            '  background: linear-gradient(135deg, #c8f542 0%, #a8e635 100%);',
            '  color: #10200a; font-weight: 700;',
            '  padding: 0.65rem 1.2rem; border-radius: 0.75rem;',
            '  font-family: Syne, "DM Sans", ui-sans-serif, system-ui, sans-serif;',
            '}',
            '.archetype-reveal .arch-cta:hover { filter: brightness(1.05); }',
            '.archetype-reveal .arch-learn-link {',
            '  font-size: 0.85rem; font-weight: 600; color: #5eead4;',
            '  text-decoration: none; border-bottom: 1px solid rgba(94,234,212,0.35);',
            '}',
            '.archetype-reveal .arch-title-link { color: inherit; text-decoration: none; }',
            '.archetype-reveal .arch-title-link:hover { color: #99f6e4; }',
            '.archetype-reveal .arch-why-label {',
            '  font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;',
            '  font-weight: 700; color: #64748b; margin-top: 0.85rem; margin-bottom: 0.35rem;',
            '}',
            '.archetype-reveal .arch-practice-heading {',
            '  font-size: 1.1rem; color: #fff; margin: 0.15rem 0 0.75rem;',
            '}',
            '.archetype-reveal .arch-drill {',
            '  position: relative;',
            '  border-radius: 0.75rem;',
            '  padding: 0.85rem 0.95rem 0.85rem 1.05rem;',
            '  background: rgba(2, 8, 16, 0.5);',
            '  border: 1px solid rgba(148,163,184,0.12);',
            '  margin-bottom: 0;',
            '  height: 100%;',
            '}',
            '.archetype-reveal .arch-drill::before {',
            '  content: ""; position: absolute; left: 0; top: 10px; bottom: 10px; width: 3px;',
            '  border-radius: 2px; background: #2dd4bf;',
            '}',
            '.archetype-reveal .arch-drill-title { font-weight: 700; color: #f8fafc; font-size: 0.95rem; }',
            '.archetype-reveal .arch-drill-why { margin-top: 0.35rem; font-size: 0.8rem; font-weight: 600; color: #5eead4; }',
            '.archetype-reveal .arch-drill-detail { margin-top: 0.4rem; font-size: 0.8rem; color: #8b96a8; line-height: 1.45; }',
            '.archetype-reveal .arch-loading-pulse {',
            '  display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 999px;',
            '  background: #c8f542; margin-right: 0.4rem;',
            '  animation: archPulse 1.1s ease-in-out infinite;',
            '}',
            '#llmCoachingPanel.llm-coach-shell {',
            '  font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif;',
            '  background: linear-gradient(165deg, #12171f 0%, #0c1016 100%);',
            '  border: 1px solid rgba(45,212,191,0.22);',
            '}',
            '#llmCoachingPanel .llm-drill {',
            '  position: relative; border-radius: 0.85rem;',
            '  padding: 0.9rem 1rem 0.9rem 1.15rem;',
            '  background: rgba(2,6,23,0.5); border: 1px solid rgba(232,236,240,0.1);',
            '  margin-bottom: 0.65rem;',
            '}',
            '#llmCoachingPanel .llm-drill::before {',
            '  content: ""; position: absolute; left: 0; top: 12px; bottom: 12px; width: 3px;',
            '  border-radius: 2px; background: #2dd4bf;',
            '}',
            '#llmCoachingPanel .llm-drill:last-child { margin-bottom: 0; }',
            '@keyframes archCardIn { to { opacity: 1; transform: translateY(0); } }',
            '@keyframes archPulse { 0%,100% { opacity: 0.35; } 50% { opacity: 1; } }',
            'body.archetype-reveal-active #resultsDetail { display: none !important; }',
            'body.archetype-reveal-active #archetypeReveal { display: block !important; }',
            'body.archetype-reveal-active #highlightReelCard { display: none !important; }',
            'body:not(.archetype-reveal-active) #archetypeReveal { display: none !important; }',
            'body.archetype-detail-active #backToStylesBtn { display: inline-flex !important; }',
            'body.archetype-reveal-active #backToStylesBtn { display: none !important; }',
        ].join('\n');
        document.head.appendChild(style);
    }

    function rewriteFencerLabels(text, userFencer, forSubject) {
        if (!text) return '';
        var out = String(text);
        var youIs1 = userFencer === 'fencer1';
        var isUser = forSubject === userFencer || !forSubject;
        var l1;
        var l2;
        if (isUser) {
            l1 = youIs1 ? 'you' : 'your opponent';
            l2 = youIs1 ? 'your opponent' : 'you';
        } else {
            l1 = forSubject === 'fencer1' ? 'they' : 'you';
            l2 = forSubject === 'fencer1' ? 'you' : 'they';
        }
        out = out.replace(/\b[Ff]encer\s*1\b/g, l1);
        out = out.replace(/\b[Ff]encer\s*2\b/g, l2);
        out = out.replace(/\bfencer1\b/g, l1);
        out = out.replace(/\bfencer2\b/g, l2);
        out = out.replace(/\bFencer1\b/g, l1);
        out = out.replace(/\bFencer2\b/g, l2);
        out = out.replace(/^you\b/, 'You').replace(/([.!?]\s+)you\b/g, '$1You');
        out = out.replace(/^they\b/, 'They').replace(/([.!?]\s+)they\b/g, '$1They');
        out = out.replace(/^your opponent\b/, 'Your opponent').replace(/([.!?]\s+)your opponent\b/g, '$1Your opponent');
        return out;
    }

    function fencerCardHtml(fencerNum, profile, opts) {
        opts = opts || {};
        var accent = fencerNum === 1 ? 'fencer1' : 'fencer2';
        var subjectKey = 'fencer' + fencerNum;
        var isYou = opts.userFencer === subjectKey;
        var name = isYou ? 'You' : ('Fencer ' + fencerNum);
        var archHref = archetypesPageHref() + '#' + (profile.archetype && profile.archetype.id ? profile.archetype.id : '');
        var rationaleRaw = rewriteFencerLabels(
            profile.rationale || '',
            opts.userFencer,
            subjectKey
        );
        var targetingRaw = rewriteFencerLabels(
            profile.targetingNotes || profile.targeting_notes || '',
            opts.userFencer,
            subjectKey
        );
        var rationale = rationaleRaw
            ? '<p class="arch-why-label">Why this fits</p>' +
              '<p class="text-sm text-slate-300 leading-relaxed">' + escapeHtml(rationaleRaw) + '</p>'
            : '';
        var targetingNote = targetingRaw
            ? '<p class="text-xs text-slate-500 mt-2">' + escapeHtml(targetingRaw) + '</p>'
            : '';
        return [
            '<div class="arch-card ' + accent + '">',
            '  <p class="arch-eyebrow">' + name + '</p>',
            '  <h3 class="arch-title text-2xl sm:text-3xl mt-2 mb-2 text-white">',
            '    <a href="' + archHref + '" class="arch-title-link">' + escapeHtml(profile.archetype.name) + '</a>',
            '  </h3>',
            '  <p class="text-sm text-slate-300/90 leading-relaxed mb-4">' + escapeHtml(profile.styleBlurb) + '</p>',
            '  <div class="arch-badge">' + escapeHtml(profile.targeting.name) + '</div>',
            rationale,
            targetingNote,
            '</div>',
        ].join('');
    }

    function escapeHtml(s) {
        return String(s || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function drillCardsHtml(suggestions, userFencer, variant) {
        if (!suggestions || !suggestions.length) return '';
        var useArch = variant === 'arch';
        return suggestions.map(function (s, idx) {
            var rank = idx + 1;
            var title = rewriteFencerLabels((s && s.title) ? s.title : String(s || ''), userFencer, userFencer);
            var why = rewriteFencerLabels((s && (s.why || s.reason)) ? (s.why || s.reason) : '', userFencer, userFencer);
            var detail = rewriteFencerLabels((s && s.detail) ? s.detail : '', userFencer, userFencer);
            var focus = rewriteFencerLabels((s && s.focus) ? s.focus : '', userFencer, userFencer);
            var rankedTitle = '#' + rank + ' · ' + title;
            if (useArch) {
                return [
                    '<div class="arch-drill">',
                    '  <p class="arch-drill-title">' + escapeHtml(rankedTitle) + '</p>',
                    why ? '  <p class="arch-drill-why">' + escapeHtml(why) + '</p>' : '',
                    detail ? '  <p class="arch-drill-detail">' + escapeHtml(detail) + '</p>' : '',
                    focus ? '  <p class="text-[11px] text-emerald-400/70 mt-2 font-semibold uppercase tracking-wide">' + escapeHtml(focus) + '</p>' : '',
                    '</div>',
                ].join('');
            }
            return [
                '<div class="llm-drill">',
                '  <p class="text-base font-extrabold text-white leading-snug">' + escapeHtml(rankedTitle) + '</p>',
                why ? '  <p class="mt-1.5 text-sm font-semibold text-emerald-400 leading-snug">' + escapeHtml(why) + '</p>' : '',
                detail ? '  <p class="mt-2 text-sm text-gray-400 leading-relaxed">' + escapeHtml(detail) + '</p>' : '',
                focus ? '  <p class="mt-2 text-[11px] font-bold uppercase tracking-wide text-emerald-400/70">' + escapeHtml(focus) + '</p>' : '',
                '</div>',
            ].join('');
        }).join('');
    }

    function practiceTipsHtml(suggestions, userFencer) {
        var cards = drillCardsHtml(suggestions, userFencer, 'arch');
        if (!cards) return '';
        return [
            '<div class="arch-card arch-practice-section">',
            '  <p class="arch-eyebrow">Top 3 for you</p>',
            '  <p class="arch-practice-heading">Recommended drills</p>',
            '  <div class="arch-drill-grid">',
            cards,
            '  </div>',
            '</div>',
        ].join('');
    }

    function coachingBodyHtml(analysis, userFencer) {
        if (!analysis) return '';
        ensureStyles();
        var you = userLlmBlock(analysis, userFencer) || analysis;
        var name = (you && you.archetype_name) || analysis.archetype_name || '';
        var style = (you && you.archetype_style) || analysis.archetype_style || '';
        var rationale = rewriteFencerLabels(
            (you && you.rationale) || analysis.rationale || '',
            userFencer,
            userFencer
        );
        var targeting = rewriteFencerLabels(
            (you && you.targeting_notes) || analysis.targeting_notes || '',
            userFencer,
            userFencer
        );
        var drills = drillCardsHtml(analysis.practice_suggestions || [], userFencer, 'panel');
        return [
            '<div class="space-y-5">',
            '  <section class="rounded-2xl border border-emerald-500/25 bg-emerald-500/[0.06] p-5">',
            '    <p class="text-[11px] uppercase tracking-[0.16em] font-bold text-emerald-400/90 mb-2">Your style</p>',
            '    <p class="llm-coach-heading text-3xl text-white">' + escapeHtml(name) + '</p>',
            style ? '    <p class="text-sm text-gray-400 mt-2 leading-relaxed">' + escapeHtml(style) + '</p>' : '',
            rationale
                ? '    <div class="mt-5 pt-4 border-t border-emerald-500/20">' +
                  '      <p class="text-[11px] uppercase tracking-[0.14em] font-bold text-gray-500 mb-2">Why this fits</p>' +
                  '      <p class="text-sm text-gray-200 leading-relaxed">' + escapeHtml(rationale) + '</p>' +
                  (targeting ? '<p class="text-xs text-gray-400 mt-3">' + escapeHtml(targeting) + '</p>' : '') +
                  '    </div>'
                : (targeting ? '<p class="text-xs text-gray-400 mt-3">' + escapeHtml(targeting) + '</p>' : ''),
            '  </section>',
            drills
                ? '  <section>' +
                  '    <div class="mb-3 pb-2 border-b border-gray-600/40">' +
                  '      <p class="text-[11px] uppercase tracking-[0.16em] font-bold text-gray-500 mb-1">Practice</p>' +
                  '      <p class="llm-coach-heading text-xl text-white">Top 3 drills</p>' +
                  '    </div>' +
                  drills +
                  '  </section>'
                : '',
            '</div>',
        ].join('');
    }

    function profileFromLlmBlock(block, fallbackTouches) {
        if (block && block.archetype_id) {
            var meta = ARCHETYPE_META[block.archetype_id];
            if (!meta) {
                return classifyFencerProfile(fallbackTouches || []);
            }
            var targeting = TARGETING_META.standard;
            if (fallbackTouches && fallbackTouches.length) {
                targeting = classifyTargeting(targetZonePcts(fallbackTouches));
            }
            return {
                archetype: meta,
                targeting: targeting,
                styleBlurb: block.archetype_style || meta.style,
                rationale: block.rationale || '',
                targetingNotes: block.targeting_notes || '',
                fromLlm: true,
            };
        }
        return classifyFencerProfile(fallbackTouches || []);
    }

    function loadingCardHtml(kind) {
        var title = kind === 'why' ? 'Writing your explanation…' : 'Writing drills…';
        var body = kind === 'why'
            ? 'Turning bout thresholds into a natural coaching note.'
            : 'Building practice recommendations for your style.';
        return [
            '<div class="arch-card">',
            '  <p class="arch-eyebrow">Coaching</p>',
            '  <h3 class="arch-title text-xl mt-2 mb-2 text-white">',
            '    <span class="arch-loading-pulse"></span>' + title,
            '  </h3>',
            '  <p class="text-sm text-slate-400 leading-relaxed">' + body + '</p>',
            '</div>',
        ].join('');
    }

    function archetypesPageHref() {
        try {
            if (typeof document !== 'undefined' && document.body && document.body.dataset.archetypesUrl) {
                return document.body.dataset.archetypesUrl;
            }
        } catch (e) { /* ignore */ }
        return '/archetypes';
    }

    function getVideoSlotEls() {
        return {
            reveal: document.getElementById('archetypeVideoHost'),
            detail: document.getElementById('resultsVideoHost'),
            video: document.getElementById('resultsVideo'),
        };
    }

    function moveVideoTo(host) {
        var els = getVideoSlotEls();
        if (!els.video || !host) return;
        if (els.video.parentElement === host) return;
        host.appendChild(els.video);
    }

    function revealInnerHtml() {
        var archetypesHref = archetypesPageHref();
        return [
            '<div class="p-4 sm:p-5 lg:p-6">',
            '  <p class="arch-hero-label">Style analysis</p>',
            '  <h2 class="arch-hero-title">Your fencing style</h2>',
            '  <p id="archLlmStatus" class="text-xs text-slate-400 mb-3 hidden"></p>',
            '  <div class="grid grid-cols-1 lg:grid-cols-5 gap-4 lg:gap-5 items-start">',
            '    <div class="lg:col-span-3 flex flex-col gap-3">',
            '      <div class="arch-video-shell" id="archetypeVideoHost"></div>',
            '      <div id="archetypeDrillsUnderVideo" class="flex flex-col gap-3"></div>',
            '    </div>',
            '    <div class="lg:col-span-2 flex flex-col gap-3" id="archetypeCards"></div>',
            '  </div>',
            '  <div class="mt-5 flex flex-wrap items-center gap-3">',
            '    <button type="button" id="viewFullAnalysisBtn" class="arch-cta">View full analysis</button>',
            '    <button type="button" id="rerunStylesLlmBtn" class="px-3 py-2 text-xs font-semibold rounded-lg border border-emerald-500/35 text-emerald-300/90 hover:bg-emerald-500/10 transition">Rerun coaching notes</button>',
            '    <a href="' + archetypesHref + '" id="archetypesLearnLink" class="arch-learn-link">Learn about archetypes →</a>',
            '  </div>',
            '</div>',
        ].join('');
    }

    function wireRevealCta(reveal) {
        var btn = reveal && reveal.querySelector('#viewFullAnalysisBtn');
        if (btn && btn.dataset.archWired !== '1') {
            btn.dataset.archWired = '1';
            btn.addEventListener('click', function () {
                showDetailedResults();
            });
        }
        var rerun = reveal && reveal.querySelector('#rerunStylesLlmBtn');
        if (rerun && rerun.dataset.archWired !== '1') {
            rerun.dataset.archWired = '1';
            rerun.addEventListener('click', function () {
                if (typeof global.rerunLlmAnalysis === 'function') {
                    global.rerunLlmAnalysis();
                }
            });
        }
    }

    function ensureRevealDom() {
        ensureStyles();
        var reveal = document.getElementById('archetypeReveal');
        var parent = document.getElementById('step-results') || document.getElementById('results-section');

        if (!reveal) {
            if (!parent) return null;
            reveal = document.createElement('div');
            reveal.id = 'archetypeReveal';
            reveal.className = 'archetype-reveal hidden mb-4';
            var detail = document.getElementById('resultsDetail');
            if (detail && detail.parentElement === parent) {
                parent.insertBefore(reveal, detail);
            } else {
                var firstCard = parent.querySelector('.bg-gray-800.rounded-2xl') || parent.firstElementChild;
                if (firstCard) parent.insertBefore(reveal, firstCard);
                else parent.appendChild(reveal);
            }
        }

        if (!reveal.querySelector('#archetypeCards') || !reveal.querySelector('#archetypeDrillsUnderVideo') || !document.getElementById('fencer-archetype-styles-v6')) {
            reveal.classList.add('archetype-reveal');
            reveal.innerHTML = revealInnerHtml();
        }
        wireRevealCta(reveal);
        return reveal;
    }

    function ensureDetailWrapper() {
        var detail = document.getElementById('resultsDetail');
        if (detail) return detail;

        var card = null;
        var step = document.getElementById('step-results');
        var section = document.getElementById('results-section');
        var root = step || section;
        if (!root) return null;

        // Prefer the main analysis card that contains resultsVideo / resultsHeader
        card = root.querySelector('#resultsHeader') && root.querySelector('#resultsHeader').closest('.bg-gray-800.rounded-2xl');
        if (!card) {
            var video = document.getElementById('resultsVideo');
            if (video) card = video.closest('.bg-gray-800.rounded-2xl');
        }
        if (!card) return null;

        detail = document.createElement('div');
        detail.id = 'resultsDetail';
        card.parentNode.insertBefore(detail, card);
        detail.appendChild(card);

        // Also move highlight reel outside detail if present before detail — leave as sibling
        return detail;
    }

    function ensureVideoHosts() {
        var video = document.getElementById('resultsVideo');
        if (!video) return;

        var detailHost = document.getElementById('resultsVideoHost');
        if (!detailHost) {
            detailHost = document.createElement('div');
            detailHost.id = 'resultsVideoHost';
            detailHost.className = 'results-video-host';
            video.parentNode.insertBefore(detailHost, video);
            detailHost.appendChild(video);
        }

        ensureRevealDom();
        var revealHost = document.getElementById('archetypeVideoHost');
        if (revealHost && !revealHost.contains(video) && !document.body.classList.contains('archetype-reveal-active')) {
            // keep video in detail host by default
            if (video.parentElement !== detailHost) detailHost.appendChild(video);
        }
    }

    function ensureBackButton() {
        var existing = document.getElementById('backToStylesBtn');
        if (existing) return existing;

        var header = document.getElementById('resultsHeader');
        if (!header) return null;
        var row = header.querySelector('.flex.flex-col') || header;
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'backToStylesBtn';
        btn.className = 'hidden mt-2 px-3 py-1.5 text-xs sm:text-sm font-semibold rounded-lg border border-emerald-500/40 text-emerald-300/90 hover:bg-emerald-500/10 transition';
        btn.textContent = '← Back to styles';
        btn.style.display = 'none';
        btn.addEventListener('click', function () {
            showArchetypeReveal();
        });

        var titleBlock = row.querySelector('div') || row;
        titleBlock.appendChild(btn);
        return btn;
    }

    function userTouchesFromOpts(fencer1Touches, fencer2Touches, userFencer) {
        if (userFencer === 'fencer2') return fencer2Touches || [];
        return fencer1Touches || [];
    }

    function hasUserLlmAnalysis(llm) {
        var analysis = llm && llm.analysis ? llm.analysis : null;
        if (!analysis) return false;
        var uf = (llm && llm.user_fencer) || global._userFencer;
        var block = null;
        if (analysis.archetype_id) {
            block = {
                archetype_id: analysis.archetype_id,
                rationale: analysis.rationale,
            };
        } else {
            block = uf === 'fencer2' ? analysis.fencer2 : analysis.fencer1;
        }
        if (!block || !block.archetype_id) return false;
        // Coaching notes incomplete until LLM rationale lands.
        if (!(block.rationale && String(block.rationale).trim())) return false;
        var tips = analysis.practice_suggestions || [];
        return tips.length >= 3;
    }

    function userLlmBlock(analysis, userFencer) {
        if (!analysis) return null;
        if (analysis.archetype_id) {
            return {
                archetype_id: analysis.archetype_id,
                archetype_name: analysis.archetype_name,
                archetype_style: analysis.archetype_style,
                rationale: analysis.rationale,
                targeting_notes: analysis.targeting_notes,
            };
        }
        if (userFencer === 'fencer2') return analysis.fencer2 || null;
        return analysis.fencer1 || null;
    }

    function renderArchetypeReveal(fencer1Touches, fencer2Touches, opts) {
        opts = opts || {};
        ensureStyles();
        ensureDetailWrapper();
        ensureVideoHosts();
        ensureRevealDom();
        ensureBackButton();

        var llm = opts.llmAnalysis || global._llmAnalysis || null;
        var analysis = llm && llm.analysis ? llm.analysis : null;
        var userFencer = opts.userFencer || (llm && llm.user_fencer) || global._userFencer || 'fencer1';
        var statusEl = document.getElementById('archLlmStatus');
        var cards = document.getElementById('archetypeCards');
        var drillsHost = document.getElementById('archetypeDrillsUnderVideo');

        var userTouches = userTouchesFromOpts(fencer1Touches, fencer2Touches, userFencer);
        var oppTouches = userFencer === 'fencer2' ? (fencer1Touches || []) : (fencer2Touches || []);
        var hasStored = hasUserLlmAnalysis(llm);
        var userBlock = hasStored ? userLlmBlock(analysis, userFencer) : null;
        var profile = hasStored
            ? profileFromLlmBlock(userBlock, userTouches)
            : classifyUserProfile(userTouches, oppTouches, userFencer);
        if (hasStored && userBlock) {
            profile.rationale = userBlock.rationale || profile.rationale || '';
            profile.targetingNotes = userBlock.targeting_notes || profile.targetingNotes || '';
        } else {
            // Keep card clean until LLM explanation arrives.
            profile.rationale = '';
            profile.targetingNotes = '';
        }
        var fencerNum = userFencer === 'fencer2' ? 2 : 1;
        var suggestions = (hasStored && analysis && analysis.practice_suggestions) || [];
        // Only show loading while a request is actually in flight — not whenever coaching is missing.
        var waitingCoach = !!opts.llmPending;
        var hasLlmWhy = !!(profile.rationale && String(profile.rationale).trim());

        if (cards) {
            cards.innerHTML = fencerCardHtml(fencerNum, profile, { userFencer: userFencer });
            if (waitingCoach && !hasLlmWhy) {
                cards.innerHTML += loadingCardHtml('why');
            }
        }
        if (drillsHost) {
            if (suggestions.length) {
                drillsHost.innerHTML = practiceTipsHtml(suggestions, userFencer);
            } else if (waitingCoach) {
                drillsHost.innerHTML = loadingCardHtml('drills');
            } else {
                drillsHost.innerHTML = '';
            }
        }
        if (statusEl) {
            statusEl.classList.remove('hidden');
            if (opts.llmError) {
                statusEl.textContent = String(opts.llmError);
            } else if (waitingCoach) {
                statusEl.textContent = 'Archetype locked from bout thresholds. Writing explanation + drills…';
            } else if (opts.llmStale) {
                statusEl.textContent = 'Bout data changed — rerun to refresh coaching notes.';
            } else if (analysis && analysis.suggestions_error) {
                statusEl.textContent = 'Archetype ready. Coaching notes failed — try Rerun coaching notes.';
            } else if (!hasStored && userFencer) {
                statusEl.textContent = 'Archetype from bout stats. Coaching notes not ready — tap Rerun coaching notes.';
            } else {
                statusEl.textContent = 'Archetype from bout stats · explanation & drills from coaching model.';
            }
        }

        global._lastArchetypeProfiles = { user: profile, fromLlm: !!hasStored };
        return { user: profile, fromLlm: !!hasStored };
    }

    function showArchetypeReveal() {
        ensureStyles();
        ensureRevealDom();
        ensureVideoHosts();
        ensureBackButton();
        document.body.classList.add('archetype-reveal-active');
        document.body.classList.remove('archetype-detail-active');
        var reveal = document.getElementById('archetypeReveal');
        if (reveal) reveal.classList.remove('hidden');
        // Hide the separate detail-page coaching panel while on the styles screen.
        var coach = document.getElementById('llmCoachingPanel');
        if (coach) coach.classList.add('hidden');
        moveVideoTo(document.getElementById('archetypeVideoHost'));
        var back = document.getElementById('backToStylesBtn');
        if (back) back.style.display = 'none';
        try {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } catch (e) {
            window.scrollTo(0, 0);
        }
    }

    function showDetailedResults() {
        ensureVideoHosts();
        document.body.classList.remove('archetype-reveal-active');
        document.body.classList.add('archetype-detail-active');
        var reveal = document.getElementById('archetypeReveal');
        if (reveal) reveal.classList.add('hidden');
        moveVideoTo(document.getElementById('resultsVideoHost'));
        var back = document.getElementById('backToStylesBtn');
        if (back) {
            back.style.display = '';
            back.classList.remove('hidden');
        }
        if (typeof global.renderLlmCoachingPanel === 'function') {
            global.renderLlmCoachingPanel();
        }
    }

    function enterRevealOrDetail(fencer1Touches, fencer2Touches, opts) {
        opts = opts || {};
        opts.fencer1Touches = fencer1Touches;
        opts.fencer2Touches = fencer2Touches;
        global._lastRevealTouches = {
            fencer1: fencer1Touches || [],
            fencer2: fencer2Touches || [],
        };

        ensureDetailWrapper();
        ensureVideoHosts();
        ensureBackButton();

        if (!shouldShowReveal(opts)) {
            document.body.classList.remove('archetype-reveal-active');
            document.body.classList.remove('archetype-detail-active');
            var reveal = document.getElementById('archetypeReveal');
            if (reveal) reveal.classList.add('hidden');
            moveVideoTo(document.getElementById('resultsVideoHost'));
            var back = document.getElementById('backToStylesBtn');
            if (back) {
                back.style.display = 'none';
                back.classList.add('hidden');
            }
            return false;
        }

        renderArchetypeReveal(fencer1Touches, fencer2Touches, opts);
        showArchetypeReveal();
        return true;
    }

    function applyLlmToReveal(llmAnalysis, opts) {
        opts = opts || {};
        var touches = global._lastRevealTouches || { fencer1: [], fencer2: [] };
        if (llmAnalysis) global._llmAnalysis = llmAnalysis;
        return renderArchetypeReveal(touches.fencer1, touches.fencer2, {
            llmAnalysis: llmAnalysis || global._llmAnalysis,
            userFencer: opts.userFencer || global._userFencer,
            llmPending: !!opts.llmPending,
            llmError: opts.llmError || null,
            llmStale: !!opts.llmStale,
        });
    }

    global.FencerArchetype = {
        classifyFencerProfile: classifyFencerProfile,
        attackActionPcts: attackActionPcts,
        targetZonePcts: targetZonePcts,
        displayAttackLabel: displayAttackLabel,
        normalizeAttackLabel: normalizeAttackLabel,
        shouldShowReveal: shouldShowReveal,
        renderArchetypeReveal: renderArchetypeReveal,
        showArchetypeReveal: showArchetypeReveal,
        showDetailedResults: showDetailedResults,
        enterRevealOrDetail: enterRevealOrDetail,
        applyLlmToReveal: applyLlmToReveal,
        coachingBodyHtml: coachingBodyHtml,
        rewriteFencerLabels: rewriteFencerLabels,
        hasUserLlmAnalysis: hasUserLlmAnalysis,
        ARCHETYPE_META: ARCHETYPE_META,
        TARGETING_META: TARGETING_META,
    };
})(typeof window !== 'undefined' ? window : this);
