--- RLBridge output module
--- Extracts and formats game state data for AI consumption

local O = {}

local last_hand_info = { handname = "None", score = 0 }

function O.get_game_state()
    local game_over = 0
    if G.STATE == G.STATES.GAME_OVER then
        game_over = 1
    end

    local blind_cleared = 0
    if G.STATE == G.STATES.ROUND_EVAL then
        blind_cleared = 1
    end

    local game_win = 0
    if G.STATE == G.STATES.ROUND_EVAL then
        local ante = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
        local blind = G.GAME and G.GAME.blind or nil
        local is_boss = blind and blind.boss or false
        if ante == 9 and is_boss then
            game_win = 1
        end
    end

    return {
        state = G.STATE,
        blind_name = G.GAME and G.GAME.blind and G.GAME.blind.name or "None",
        blind_cleared = blind_cleared,
        game_win = game_win,
        ante = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 1,
        round_count = G.GAME and G.GAME.round or 0,
        blind_is_boss = (G.GAME and G.GAME.blind and G.GAME.blind.boss) and 1 or 0,
        game_over = game_over,
        round = O.get_round_info(),
        blind_chips = O.get_blind_chips(),
        hand = O.get_hand_info(),
        chips = G.GAME and G.GAME.chips or 0,
        current_hand = O.get_current_hand_scoring(),
        seed = G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed or 0,
        gold = O.get_gold(),
        reroll_cost = G.GAME and G.GAME.current_round and G.GAME.current_round.reroll_cost or 5,
        deck_name = G.GAME and G.GAME.selected_back and G.GAME.selected_back.name or "Red",
        jokers = O.get_jokers(),
        consumables = O.get_consumables(),
        pack_cards = O.get_pack_cards(),
        pack_choices = G.GAME and G.GAME.pack_choices or 0,
        deck_cards = O.get_deck_cards(),
        auto_endless_config = auto_endless,
        shop = { items = O.get_shop_items() },
    }
end

function O.get_blind_chips()
    if not G.GAME or not G.GAME.blind then
        return 300
    end
    local ok, chips = pcall(function()
        return G.GAME.blind.chips
    end)
    return (ok and chips) or 300
end

function O.get_hand_info()
    if not G.hand or not G.hand.cards then
        return { cards = {}, size = 0, highlighted_count = 0 }
    end

    local hand_cards = {}
    for _, card in ipairs(G.hand.cards) do
        table.insert(hand_cards, {
            base = {
                nominal = card.base.nominal or 0,
                value = card.base.value or ""
            },
            highlighted = card.highlighted or false,
            suit = card.base.suit or ""
        })
    end

    return {
        cards = hand_cards,
        size = #hand_cards,
        highlighted_count = G.hand.highlighted and #G.hand.highlighted or 0
    }
end

function O.get_round_info()
    return {
        hands_left = G.GAME and G.GAME.current_round and G.GAME.current_round.hands_left or 0,
        discards_left = G.GAME and G.GAME.current_round and G.GAME.current_round.discards_left or 0,
    }
end

function O.get_current_hand_scoring()
    local chips = 0
    local mult = 0
    local score = 0
    local handname = "None"

    if G.GAME and G.GAME.current_round and G.GAME.current_round.current_hand then
        chips = G.GAME.current_round.current_hand.chips or 0
        mult = G.GAME.current_round.current_hand.mult or 0
        score = G.GAME.current_round.current_hand.chip_total or (chips * mult)
        handname = G.GAME.current_round.current_hand.handname or "None"
    end

    if handname and handname ~= "None" and handname ~= "" then
        last_hand_info.handname = handname
        last_hand_info.score = score
    end

    return { chips = chips, mult = mult, score = score, handname = handname,
             last_handname = last_hand_info.handname, last_score = last_hand_info.score }
end

function O.get_gold()
    return G.GAME and G.GAME.dollars or 0
end

local function get_edition_str(card)
    if card.edition then
        if card.edition.foil then return "foil" end
        if card.edition.holo then return "holo" end
        if card.edition.polychrome then return "polychrome" end
        if card.edition.negative then return "negative" end
    end
    return "none"
end

function O.get_jokers()
    local jokers = {}
    if G.jokers and G.jokers.cards then
        for i, card in ipairs(G.jokers.cards) do
            table.insert(jokers, {
                name = card.config.center.name or "Unknown",
                sell_cost = card.sell_cost or 1,
                slot = i,
                edition = get_edition_str(card)
            })
        end
    end
    return jokers
end

function O.get_consumables()
    local consumables = {}
    if G.consumeables and G.consumeables.cards then
        for i, card in ipairs(G.consumeables.cards) do
            table.insert(consumables, {
                name = card.config.center.name or "Unknown",
                item_set = (card.config.center.set or "Unknown"),
                slot = i
            })
        end
    end
    return consumables
end

function O.get_pack_cards()
    local pack_cards = {}
    if G.pack_cards and G.pack_cards.cards then
        for i, card in ipairs(G.pack_cards.cards) do
            local name = "Unknown"
            local item_set = "Unknown"
            if card.config and card.config.center then
                name = card.config.center.name or "Unknown"
                item_set = card.config.center.set or "Unknown"
            elseif card.base then
                name = (card.base.value or "") .. (card.base.suit or "")
                item_set = "Standard"
            end
            table.insert(pack_cards, {
                name = name,
                item_set = item_set,
                slot = i
            })
        end
    end
    return pack_cards
end

function O.get_deck_cards()
    local deck_info = {}
    if G.playing_cards then
        for _, card in ipairs(G.playing_cards) do
            local seal = card.seal or "None"
            local enhancement = (card.ability and card.ability.name) or "Base"
            table.insert(deck_info, {
                seal = seal,
                enhancement = enhancement,
                edition = get_edition_str(card)
            })
        end
    end
    return deck_info
end

function O.get_shop_items()
    local items = {}

    if G.shop_jokers and G.shop_jokers.cards then
        for i, card in ipairs(G.shop_jokers.cards) do
            table.insert(items, {
                name = card.config.center.name or "Unknown",
                cost = card.cost or 999,
                type = "joker",
                item_set = card.config.center.set or "Joker",
                slot = i
            })
        end
    end

    if G.shop_vouchers and G.shop_vouchers.cards then
        for i, card in ipairs(G.shop_vouchers.cards) do
            table.insert(items, {
                name = card.config.center.name or "Unknown",
                cost = card.cost or 10,
                type = "voucher",
                item_set = "Voucher",
                slot = i
            })
        end
    end

    if G.shop_booster and G.shop_booster.cards then
        for i, card in ipairs(G.shop_booster.cards) do
            local pack_size = 0
            local pack_choose = 1
            if card.ability then pack_size = card.ability.extra or 0 end
            if card.config and card.config.center and card.config.center.config then
                pack_choose = card.config.center.config.choose or 1
            end
            table.insert(items, {
                name = card.config.center.name or "Unknown",
                cost = card.cost or 4,
                type = "booster",
                item_set = "Booster",
                slot = i,
                pack_size = pack_size,
                pack_choose = pack_choose
            })
        end
    end

    return items
end

return O
