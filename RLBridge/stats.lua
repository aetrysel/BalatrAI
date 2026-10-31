--- RLBridge stats module
--- Read-only hooks that count per-run events the game doesn't track itself (tags acquired, booster packs, boss rerolls)
--- Counters live on G.GAME.rl_stats so they reset with every new run; exported via output.get_run_stats()

local STATS = {}

local installed = false
local tag_source_override = nil  -- set while a known source (blind skip) is adding a tag

local function run_stats()
    if not G.GAME then
        return nil
    end
    G.GAME.rl_stats = G.GAME.rl_stats or {
        tags = { by_key = {}, by_source = {} },
        packs = {},
        boss_rerolls = 0,
        pending_double = {},  -- tag key → copies Double Tags have promised but not yet added
    }
    return G.GAME.rl_stats
end

local function incr(tbl, key, amount)
    tbl[key] = (tbl[key] or 0) + (amount or 1)
end

local function pack_entry_by_name(stats, name)
    stats.packs[name] = stats.packs[name] or { offered = 0, opened = 0, bought = 0, picks = 0, skips = 0 }
    return stats.packs[name], name
end

local function pack_entry(stats, card)
    local name = card and card.config and card.config.center and card.config.center.name or "Unknown"
    return pack_entry_by_name(stats, name)
end

--- Attribute a tag that is about to be added
--- double: a copy fired by a Double Tag; skip: added by skipping a blind;
--- anaglyph: Double Tag from Anaglyph Deck after a boss; diet_cola: Double Tag from selling Diet Cola
local function tag_source(stats, key)
    if (stats.pending_double[key] or 0) > 0 then
        stats.pending_double[key] = stats.pending_double[key] - 1
        return "double"
    end
    if tag_source_override then
        return tag_source_override
    end
    if key == "tag_double" then
        local deck = G.GAME.selected_back and G.GAME.selected_back.name
        if deck == "Anaglyph Deck" and G.STATE ~= G.STATES.SHOP and G.STATE ~= G.STATES.SELECTING_HAND then
            return "anaglyph"
        end
        return "diet_cola"
    end
    return "other"
end

--- Wrap game functions once (call after G:start_up so G.FUNCS is populated)
function STATS.install()
    if installed then
        return
    end
    installed = true

    local original_add_tag = add_tag
    add_tag = function(_tag)
        local stats = run_stats()
        if stats and _tag and _tag.key then
            local key = _tag.key
            incr(stats.tags.by_key, key)
            incr(stats.tags.by_source, tag_source(stats, key))
            -- Each untriggered Double Tag will add one copy of this tag (tag.lua tag_add context)
            if key ~= "tag_double" then
                local doubles = 0
                for _, t in ipairs(G.GAME.tags or {}) do
                    if t.name == "Double Tag" and not t.triggered then
                        doubles = doubles + 1
                    end
                end
                if doubles > 0 then
                    incr(stats.pending_double, key, doubles)
                end
            end
        end
        return original_add_tag(_tag)
    end

    local original_skip_blind = G.FUNCS.skip_blind
    G.FUNCS.skip_blind = function(e)
        tag_source_override = "skip"
        local ok, err = pcall(original_skip_blind, e)
        tag_source_override = nil
        if not ok then
            error(err)
        end
    end

    local original_open = Card.open
    Card.open = function(self, ...)
        local stats = run_stats()
        if stats and self.ability and self.ability.set == "Booster" then
            local entry = pack_entry(stats, self)
            entry.opened = entry.opened + 1
            if not self.from_tag then
                entry.bought = entry.bought + 1
            end
            stats.current_pack = self.config.center.name
        end
        return original_open(self, ...)
    end

    local original_use_card = G.FUNCS.use_card
    G.FUNCS.use_card = function(e, ...)
        local stats = run_stats()
        local card = e and e.config and e.config.ref_table
        if stats and card and G.pack_cards and card.area == G.pack_cards and stats.current_pack then
            local entry = stats.packs[stats.current_pack]
            if entry then
                entry.picks = entry.picks + 1
            end
        end
        return original_use_card(e, ...)
    end

    -- Shop booster slots are rolled through get_pack('shop_pack') (only caller in the game)
    local original_get_pack = get_pack
    get_pack = function(_key, ...)
        local center = original_get_pack(_key, ...)
        local stats = run_stats()
        if stats and _key == 'shop_pack' and center and center.name then
            local entry = pack_entry_by_name(stats, center.name)
            entry.offered = (entry.offered or 0) + 1
        end
        return center
    end

    local original_reroll_boss = G.FUNCS.reroll_boss
    G.FUNCS.reroll_boss = function(e, ...)
        local stats = run_stats()
        if stats then
            stats.boss_rerolls = (stats.boss_rerolls or 0) + 1
        end
        return original_reroll_boss(e, ...)
    end

    local original_skip_booster = G.FUNCS.skip_booster
    G.FUNCS.skip_booster = function(e, ...)
        local stats = run_stats()
        if stats and stats.current_pack and stats.packs[stats.current_pack] then
            local entry = stats.packs[stats.current_pack]
            entry.skips = entry.skips + 1
        end
        return original_skip_booster(e, ...)
    end
end

--- Per-run tag and booster counters for output.get_run_stats()
--- @return table {tags = {by_key, by_source}, packs = {[name] = {offered, opened, bought, picks, skips}}, boss_rerolls}
function STATS.get()
    local stats = run_stats()
    if not stats then
        return { tags = { by_key = {}, by_source = {} }, packs = {}, boss_rerolls = 0 }
    end
    return { tags = stats.tags, packs = stats.packs, boss_rerolls = stats.boss_rerolls or 0 }
end

return STATS
