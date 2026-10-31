--- RLBridge joker tracker module
--- Read-only hooks that measure what each owned joker actually does:
---   score share   — fraction of each hand's final score (chips × mult) that depends on the joker's effects
---   value         — money and consumables/jokers the joker generated, per round
---   upgrades      — permanent increases to the joker's own scaling fields (metrics only)
--- State lives on G.GAME.rl_jokers (keyed by tostring(sort_id)) so it resets with every new run

local JT = {}

local SCORE_EMA_ALPHA = 0.3
local VALUE_EMA_ALPHA = 0.3
local VALUE_NORM = 8.0          -- $ + 3 × cards created per round that counts as full value
local CARD_VALUE_DOLLARS = 3.0  -- dollar-equivalent of one generated consumable/joker

local installed = false
local depth = 0             -- calculate_joker nesting depth (Blueprint/Brainstorm call the copied joker)
local current = nil         -- tracker entry of the outermost joker being calculated
local money_seen = false    -- ease_dollars fired during the current outermost call
local scoring = nil         -- per-hand accumulator while G.FUNCS.evaluate_play runs

local function store()
    if not G.GAME then
        return nil
    end
    G.GAME.rl_jokers = G.GAME.rl_jokers or { by_id = {}, upgrades_total = 0 }
    return G.GAME.rl_jokers
end

local function is_owned_joker(card)
    return card and card.ability and card.ability.set == "Joker" and G.jokers and card.area == G.jokers
end

local function entry_for(card)
    local s = store()
    if not s or not card.sort_id then
        return nil
    end
    local key = tostring(card.sort_id)
    local e = s.by_id[key]
    if not e then
        e = {
            score_ema = 0, value_ema = 0, triggered_last_hand = false,
            round_dollars = 0, round_cards = 0,
            dollars_generated = 0, cards_created = 0, upgrades = 0, hands_scored = 0,
        }
        s.by_id[key] = e
    end
    return e, key
end

local function log_gain(new, old)
    if old <= 0 or new <= old then
        return 0
    end
    return math.log(new / old)
end

--- Log-space contribution of one scoring effect, measured against the running hand_chips/mult
local function effect_contribution(ret, n_scoring)
    local hc = math.max(tonumber(hand_chips) or 0, 1)
    local m = math.max(tonumber(mult) or 0, 1)
    local c = 0
    local extra = type(ret.extra) == "table" and ret.extra or {}
    local chips = ret.chip_mod or ret.chips or extra.chip_mod
    if type(chips) == "number" and chips > 0 then
        c = c + log_gain(hc + chips, hc)
    end
    local add_mult = ret.mult_mod or ret.mult or ret.h_mult or extra.mult_mod
    if type(add_mult) == "number" and add_mult > 0 then
        c = c + log_gain(m + add_mult, m)
    end
    -- xmult/Xmult/x_chips are Steamodded return keys
    local x = ret.Xmult_mod or ret.x_mult or ret.x_mult_mod or ret.xmult or ret.Xmult
    if type(x) == "number" and x > 1 then
        c = c + math.log(x)
    end
    local xc = ret.x_chips or ret.xchips
    if type(xc) == "number" and xc > 1 then
        c = c + math.log(xc)
    end
    local reps = ret.repetitions
    if type(reps) == "number" and reps > 0 then
        -- A retrigger is worth about one more scoring card
        c = c + reps * math.log(1 + 1 / math.max(n_scoring or 5, 1))
    end
    return c
end

local function credit_score(key, amount)
    if scoring and amount > 0 then
        scoring.contrib[key] = (scoring.contrib[key] or 0) + amount
    end
end

local function credit_money(entry, amount)
    if entry and type(amount) == "number" and amount > 0 then
        entry.round_dollars = entry.round_dollars + amount
        entry.dollars_generated = entry.dollars_generated + amount
    end
end

local function credit_cards(entry, count)
    if entry and count > 0 then
        entry.round_cards = entry.round_cards + count
        entry.cards_created = entry.cards_created + count
    end
end

-- Scaling fields jokers permanently upgrade (Green Joker, Hologram, Runner, Lucky Cat, Caino, Egg, Rocket...)
local function snapshot(ability)
    local extra = type(ability.extra) == "table" and ability.extra or {}
    return {
        ability.mult, ability.x_mult, ability.caino_xmult, ability.extra_value,
        extra.chips, extra.mult, extra.Xmult, extra.x_mult, extra.dollars,
    }
end

local function count_upgrades(before, ability)
    local after = snapshot(ability)
    local n = 0
    for i = 1, 9 do
        if type(before[i]) == "number" and type(after[i]) == "number" and after[i] > before[i] then
            n = n + 1
        end
    end
    return n
end

local function pack(...)
    return { n = select("#", ...), ... }
end

local function is_scoring_context(context)
    return context.joker_main or context.individual or context.other_joker or context.repetition
end

--- Wrap game functions once (call after G:start_up so G.FUNCS is populated)
function JT.install()
    if installed then
        return
    end
    installed = true

    local original_calculate_joker = Card.calculate_joker
    Card.calculate_joker = function(self, context)
        if depth > 0 or not is_owned_joker(self) then
            return original_calculate_joker(self, context)
        end
        local entry, key = entry_for(self)
        if not entry then
            return original_calculate_joker(self, context)
        end

        local cbuf = G.GAME.consumeable_buffer or 0
        local jbuf = G.GAME.joker_buffer or 0
        local before = (context and not context.blueprint) and snapshot(self.ability) or nil
        depth, current, money_seen = 1, entry, false
        local res = pack(original_calculate_joker(self, context))
        local saw_money = money_seen
        depth, current, money_seen = 0, nil, false

        -- Consumables/jokers queued synchronously by this joker (Superposition, Cartomancer, Riff-Raff...)
        credit_cards(entry, math.max(0, (G.GAME.consumeable_buffer or 0) - cbuf)
            + math.max(0, (G.GAME.joker_buffer or 0) - jbuf))

        if before then
            local ups = count_upgrades(before, self.ability)
            if ups > 0 then
                entry.upgrades = entry.upgrades + ups
                local s = store()
                s.upgrades_total = (s.upgrades_total or 0) + ups
            end
        end

        local ret = res[1]
        if type(ret) == "table" then
            if scoring and context and is_scoring_context(context) then
                local n_scoring = context.scoring_hand and #context.scoring_hand or 5
                credit_score(key, effect_contribution(ret, n_scoring))
            end
            -- Returned dollars are paid by evaluate_play (Golden Ticket, Business Card, Rough Gem, Reserved Parking)
            if scoring and not saw_money then
                credit_money(entry, ret.dollars)
            end
            -- 8 Ball creates its Tarot inside extra.func, which evaluate_play calls later
            if type(ret.extra) == "table" and type(ret.extra.func) == "function" then
                local original_func = ret.extra.func
                ret.extra.func = function(...)
                    local buf = G.GAME.consumeable_buffer or 0
                    local out = pack(original_func(...))
                    credit_cards(entry, math.max(0, (G.GAME.consumeable_buffer or 0) - buf))
                    return unpack(out, 1, out.n)
                end
            end
        end
        return unpack(res, 1, res.n)
    end

    local original_get_edition = Card.get_edition
    Card.get_edition = function(self, ...)
        local ret = original_get_edition(self, ...)
        if scoring and type(ret) == "table" and is_owned_joker(self) then
            local _, key = entry_for(self)
            if key then
                credit_score(key, effect_contribution(ret, 5))
            end
        end
        return ret
    end

    -- Money paid directly inside a joker's calculation (Matador, Trading Card, Mail-In Rebate, To Do List...)
    local original_ease_dollars = ease_dollars
    ease_dollars = function(mod, ...)
        if current and type(mod) == "number" and mod > 0 then
            money_seen = true
            credit_money(current, mod)
        end
        return original_ease_dollars(mod, ...)
    end

    -- End-of-round payouts (Golden Joker, Cloud 9, Rocket, Satellite, Delayed Gratification)
    local original_dollar_bonus = Card.calculate_dollar_bonus
    Card.calculate_dollar_bonus = function(self, ...)
        local ret = original_dollar_bonus(self, ...)
        if type(ret) == "number" and ret > 0 and is_owned_joker(self) then
            credit_money((entry_for(self)), ret)
        end
        return ret
    end

    local original_evaluate_play = G.FUNCS.evaluate_play
    G.FUNCS.evaluate_play = function(e, ...)
        scoring = { contrib = {} }
        local ok, err = pcall(original_evaluate_play, e, ...)
        local hand = scoring
        scoring = nil
        if not ok then
            error(err)
        end
        JT.finish_hand(hand.contrib)
    end

    -- A new blind closes the previous round's value window
    local original_new_round = new_round
    new_round = function(...)
        JT.finish_round()
        return original_new_round(...)
    end
end

--- Fold one hand's per-joker contributions into each owned joker's score EMA
--- Share = 1 − e^(−c): fraction of the hand's score that would be lost without the joker
--- (×3 → 0.67, ×1.5 → 0.33, +4 mult on 4 → 0.5, +4 mult on 60 → 0.06)
--- @param contrib table key → summed log contribution this hand
function JT.finish_hand(contrib)
    if not G.jokers or not G.jokers.cards then
        return
    end
    if (tonumber(hand_chips) or 0) * (tonumber(mult) or 0) <= 0 then
        return  -- hand scored nothing (boss-blocked); no information about the jokers
    end
    for _, card in ipairs(G.jokers.cards) do
        local entry, key = entry_for(card)
        if entry then
            local share = 1 - math.exp(-(contrib[key] or 0))
            entry.score_ema = (1 - SCORE_EMA_ALPHA) * entry.score_ema + SCORE_EMA_ALPHA * share
            entry.triggered_last_hand = share > 0
            entry.hands_scored = entry.hands_scored + 1
        end
    end
end

--- Fold each owned joker's money/cards generated this round into its value EMA
function JT.finish_round()
    if not G.jokers or not G.jokers.cards then
        return
    end
    for _, card in ipairs(G.jokers.cards) do
        local entry = entry_for(card)
        if entry then
            local value = math.min(1, (entry.round_dollars + CARD_VALUE_DOLLARS * entry.round_cards) / VALUE_NORM)
            entry.value_ema = (1 - VALUE_EMA_ALPHA) * entry.value_ema + VALUE_EMA_ALPHA * value
            entry.round_dollars = 0
            entry.round_cards = 0
        end
    end
end

--- Tracker fields for one joker card, merged into output.get_jokers()
--- @return table {contribution, score_ema, value_ema, triggered_last_hand, upgrades, dollars_generated, cards_created}
function JT.get(card)
    local entry = card and card.sort_id and store() and entry_for(card)
    if not entry then
        return { contribution = 0, score_ema = 0, value_ema = 0, triggered_last_hand = false,
                 upgrades = 0, dollars_generated = 0, cards_created = 0 }
    end
    return {
        contribution = math.min(1, entry.score_ema + entry.value_ema),
        score_ema = entry.score_ema,
        value_ema = entry.value_ema,
        triggered_last_hand = entry.triggered_last_hand,
        upgrades = entry.upgrades,
        dollars_generated = entry.dollars_generated,
        cards_created = entry.cards_created,
    }
end

--- Total joker upgrade events this run
function JT.upgrades_total()
    local s = store()
    return s and s.upgrades_total or 0
end

return JT
