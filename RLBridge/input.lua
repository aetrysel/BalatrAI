--- RLBridge input module
--- Handles direct game input actions like selecting blinds and making plays
--- Provides interface between AI actions and Balatro's internal functions

local I = {}
local utils = require("utils")

--- Start the basic run
--- Automatically starts the run in the main menu
--- @param seed_choice optional override seed value (string or number)
--- @return table Result with success status and optional error message
function I.start_run(params)
    local _seed = (params and params.seed) or (G.run_setup_seed and G.setup_seed) or G.forced_seed or nil
    local _challenge = G.challenge_tab or nil
    local _stake = G.forced_stake or G.PROFILES[G.SETTINGS.profile].MEMORY.stake or 1

    if _seed then
        G.forced_seed = _seed
        G.SETTINGS.seed = _seed
        G.run_setup_seed = _seed
        G.setup_seed = _seed
    end

    if params and params.deck and G.GAME then
        local deck_center = get_deck_from_name(params.deck)
        if deck_center then
            G.GAME.selected_back = Back(deck_center)
        end
    end

    G.FUNCS.start_run(nil, {
        stake = _stake,
        seed = _seed,
        challenge = _challenge
    })
    
    utils.log_input("start_run " .. utils.completed_success_msg)
    return { success = true }
end
---Comment Out when not debugging
function I.force_beat_blind()
    if G.GAME and G.GAME.chips ~= nil then
        G.GAME.chips = 999999999
    end
    utils.log_input("force_beat_blind: set chips to 999999999")
    return { success = true }
end

--- Select the current blind on deck
--- Automatically selects the next available blind in the blind selection screen
--- @return table Result with success status and optional error message
function I.select_blind()
    local blind_on_deck = G.GAME.blind_on_deck
    local blind_key = G.GAME.round_resets.blind_choices[blind_on_deck]

    -- Replicate select_blind internals without UIBox traversal
    stop_use()
    G.GAME.facing_blind = true

    G.E_MANAGER:add_event(Event({
        trigger = 'after',
        func = function()
            ease_round(1)
            inc_career_stat('c_rounds', 1)
            G.GAME.round_resets.blind = G.P_BLINDS[blind_key]
            G.GAME.round_resets.blind_states[blind_on_deck] = 'Current'
            if G.blind_select then
                G.blind_select:remove()
                G.blind_select = nil
            end
            if G.blind_prompt_box then
                G.blind_prompt_box:remove()
                G.blind_prompt_box = nil
            end
            delay(0.2)
            return true
        end
    }))

    G.E_MANAGER:add_event(Event({
        trigger = 'after',
        func = function()
            new_round()
            return true
        end
    }))

    utils.log_input("select_blind " .. utils.completed_success_msg)
    return { success = true }
end

--- Skip the current blind on deck (Small or Big only)
--- @return table Result with success status and optional error message
function I.skip_blind()
    local blind_on_deck = G.GAME.blind_on_deck
    if not blind_on_deck or blind_on_deck == 'Boss' then
        return { success = false, error = "Cannot skip Boss blind" }
    end

    stop_use()
    G.GAME.skips = (G.GAME.skips or 0) + 1

    local tag_key = G.GAME.round_resets.blind_tags
                    and G.GAME.round_resets.blind_tags[blind_on_deck]
    if tag_key then
        local tag = Tag(tag_key, nil, blind_on_deck)
        if tag then add_tag(tag) end
    end

    local skip_to = blind_on_deck == 'Small' and 'Big' or 'Boss'
    G.GAME.round_resets.blind_states[blind_on_deck] = 'Skipped'
    G.GAME.round_resets.blind_states[skip_to] = 'Select'
    G.GAME.blind_on_deck = skip_to

    play_sound('generic1')

    G.E_MANAGER:add_event(Event({
        trigger = 'immediate',
        func = function()
            delay(0.3)
            for i = 1, #G.jokers.cards do
                G.jokers.cards[i]:calculate_joker({skip_blind = true})
            end
            save_run()
            for i = 1, #G.GAME.tags do
                G.GAME.tags[i]:apply_to_run({type = 'immediate'})
            end
            for i = 1, #G.GAME.tags do
                if G.GAME.tags[i]:apply_to_run({type = 'new_blind_choice'}) then break end
            end
            return true
        end
    }))

    utils.log_input("skip_blind " .. utils.completed_success_msg)
    return { success = true }
end

--- Select a hand
--- Selects the cards based on a table of indexes
--- @param card_indices table Array of card indices to select
--- @return table Result with success status and optional error message
function I.select_hand(card_indices, boss_name)
    if not card_indices or type(card_indices) ~= "table" then
        return { success = false, error = "Invalid card indices parameter" }
    end

    if #card_indices < 1 then
        return { success = false, error = "Must play minimum one card" }
    end
    if #card_indices > 5 then
        return { success = false, error = "Must play maximum 5 cards" }
    end

    -- Validate hand exists and has cards
    if not G.hand or not G.hand.cards or #G.hand.cards == 0 then
        return { success = false, error = "No hand or cards available" }
    end

    local handsize = #G.hand.cards;

    -- Validate all card indices are within bounds
    for i = 1, #card_indices do
        local card_index = card_indices[i]
        if not card_index or card_index < 1 or card_index > handsize then
            return { success = false, error = "Card index out of bounds: " .. tostring(card_index) }
        end
        if not G.hand.cards[card_index] then
            return { success = false, error = "Card at index " .. card_index .. " does not exist" }
        end
    end

    -- Click the cards
    for i = 1, #card_indices do
        G.hand.cards[card_indices[i]]:click()
    end
    utils.log_input("select_hand " .. utils.completed_success_msg)
    return { success = true }
end

--- Click "Play Hand" button to play a hand that was selected
--- @return table Result with success status and optional error message
function I.play_hand()
    utils.log_input("play_hand " .. utils.completed_success_msg)
    G.FUNCS.play_cards_from_highlighted()
    return { success = true }
end

--- Click "Discard" button to discard a hand that was selected
--- @return table Result with success status and optional error message
function I.discard_hand()
    G.FUNCS.discard_cards_from_highlighted()
    utils.log_input("select_hand " .. utils.completed_success_msg)
    return { success = true }
end
--- Buy any item from G.shop_jokers by slot (jokers, tarots, planets that appear there)
function I.buy_shop_joker_slot(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end
    if not G.shop_jokers or not G.shop_jokers.cards then
        return { success = false, error = "Shop joker area not available" }
    end
    if slot < 1 or slot > #G.shop_jokers.cards then
        return { success = false, error = "Slot index out of bounds: " .. tostring(slot) }
    end
    local card = G.shop_jokers.cards[slot]
    if not card then
        return { success = false, error = "No card in slot: " .. tostring(slot) }
    end
    if G.GAME.dollars < card.cost then
        return { success = false, error = "Cannot afford card, cost: " .. tostring(card.cost) }
    end
    G.FUNCS.buy_from_shop({ config = { ref_table = card } })
    utils.log_input("buy_shop_joker_slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true }
end

--- Buy the voucher from G.shop_vouchers slot 1
function I.buy_voucher()
    if not G.shop_vouchers or not G.shop_vouchers.cards or #G.shop_vouchers.cards == 0 then
        return { success = false, error = "No voucher available" }
    end
    local card = G.shop_vouchers.cards[1]
    if not card then
        return { success = false, error = "No voucher card found" }
    end
    if G.GAME.dollars < (card.cost or 10) then
        return { success = false, error = "Cannot afford voucher, cost: " .. tostring(card.cost) }
    end
    local ok, err = pcall(function()
        G.FUNCS.use_card({ config = { ref_table = card } })
    end)
    if not ok then
        utils.log_input("buy_voucher ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end
    utils.log_input("buy_voucher " .. utils.completed_success_msg)
    return { success = true }
end

--- Buy a booster pack from G.shop_booster by slot index
function I.buy_booster(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end
    if not G.shop_booster or not G.shop_booster.cards then
        return { success = false, error = "Shop booster area not available" }
    end
    if slot < 1 or slot > #G.shop_booster.cards then
        return { success = false, error = "Booster slot out of bounds: " .. tostring(slot) }
    end
    local card = G.shop_booster.cards[slot]
    if not card then
        return { success = false, error = "No booster in slot: " .. tostring(slot) }
    end
    if G.GAME.dollars < (card.cost or 4) then
        return { success = false, error = "Cannot afford booster, cost: " .. tostring(card.cost) }
    end
    local ok, err = pcall(function()
        G.FUNCS.use_card({ config = { ref_table = card } })
    end)
    if not ok then
        utils.log_input("buy_booster ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end
    utils.log_input("buy_booster slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true }
end

--- Buy and immediately use a consumable (tarot/planet) from G.shop_jokers by slot index
function I.buy_and_use_shop_item(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end
    if not G.shop_jokers or not G.shop_jokers.cards then
        return { success = false, error = "Shop joker area not available" }
    end
    if slot < 1 or slot > #G.shop_jokers.cards then
        return { success = false, error = "Slot index out of bounds: " .. tostring(slot) }
    end
    local card = G.shop_jokers.cards[slot]
    if not card then
        return { success = false, error = "No card in slot: " .. tostring(slot) }
    end
    if not (card.ability and card.ability.consumeable) then
        return { success = false, error = "Not a consumable in slot: " .. tostring(slot) }
    end
    if G.GAME.dollars < (card.cost or 3) then
        return { success = false, error = "Cannot afford, cost: " .. tostring(card.cost) }
    end
    local ok, err = pcall(function()
        G.FUNCS.buy_from_shop({ config = { ref_table = card, id = 'buy_and_use' } })
    end)
    if not ok then
        utils.log_input("buy_and_use_shop_item ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end
    utils.log_input("buy_and_use_shop_item slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true }
end

--- Sell a consumable from G.consumeables by slot index
function I.sell_consumable(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end
    if not G.consumeables or not G.consumeables.cards or #G.consumeables.cards == 0 then
        return { success = false, error = "No consumables available" }
    end
    if slot < 1 or slot > #G.consumeables.cards then
        return { success = false, error = "Consumable slot out of bounds: " .. tostring(slot) }
    end
    local card = G.consumeables.cards[slot]
    if not card then
        return { success = false, error = "No consumable in slot: " .. tostring(slot) }
    end
    card:sell_card()
    utils.log_input("sell_consumable slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true }
end

--- Use a consumable card from G.consumeables by slot index
--- Hand cards should already be highlighted (via select_hand) if the consumable needs targets
function I.use_consumable(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end
    if not G.consumeables or not G.consumeables.cards or #G.consumeables.cards == 0 then
        return { success = false, error = "No consumables available" }
    end
    if slot < 1 or slot > #G.consumeables.cards then
        return { success = false, error = "Consumable slot out of bounds: " .. tostring(slot) }
    end
    local card = G.consumeables.cards[slot]
    if not card then
        return { success = false, error = "No consumable in slot: " .. tostring(slot) }
    end
    if not card:can_use_consumeable() then
        return { success = false, error = "Consumable cannot be used right now" }
    end
    local ok, err = pcall(function()
        G.FUNCS.use_card({ config = { ref_table = card } })
    end)
    if not ok then
        utils.log_input("use_consumable ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end
    utils.log_input("use_consumable slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true }
end

--- Select a card from G.pack_cards by slot index (during booster pack opening)
function I.select_pack_card(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end
    if not G.pack_cards or not G.pack_cards.cards or #G.pack_cards.cards == 0 then
        return { success = false, error = "No pack cards available" }
    end
    if slot < 1 or slot > #G.pack_cards.cards then
        return { success = false, error = "Pack card slot out of bounds: " .. tostring(slot) }
    end
    local card = G.pack_cards.cards[slot]
    if not card then
        return { success = false, error = "No pack card in slot: " .. tostring(slot) }
    end
    -- Some spectrals (Deja Vu, Trance, Medium, Talisman, Aura, Cryptid, etc.) require highlighted
    -- hand cards. use_consumeable captures G.hand.highlighted[1] synchronously; if nil, the
    -- deferred set_seal event crashes outside our pcall.
    local min_cards = card.ability and card.ability.min_cards or 0
    if min_cards > 0 then
        local highlighted = G.hand and G.hand.highlighted or {}
        if #highlighted < min_cards then
            return { success = false, error = "Card requires " .. min_cards .. " highlighted hand card(s); currently " .. #highlighted }
        end
    end
    local ok, err = pcall(function()
        G.FUNCS.use_card({ config = { ref_table = card } })
    end)
    if not ok then
        utils.log_input("select_pack_card ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end
    utils.log_input("select_pack_card slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true }
end

--- Skip remaining pack card selections and close the booster pack
function I.skip_booster()
    local ok, err = pcall(function()
        G.FUNCS.skip_booster({})
    end)
    if not ok then
        utils.log_input("skip_booster ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end
    utils.log_input("skip_booster " .. utils.completed_success_msg)
    return { success = true }
end

---For selling a joker
function I.sell_joker(slot)
    if not slot or type(slot) ~= "number" then
        return { success = false, error = "Invalid slot parameter" }
    end

    if not G.jokers or not G.jokers.cards then
        return { success = false, error = "No jokers available" }
    end

    if slot < 1 or slot > #G.jokers.cards then
        return { success = false, error = "Joker slot out of bounds: " .. tostring(slot) }
    end

    local card = G.jokers.cards[slot]
    if not card then
        return {success = false, error = "No joker in slot: " .. tostring(slot) }
    end

    card:sell_card()
    utils.log_input("sell_joker slot " .. tostring(slot) .. " " .. utils.completed_success_msg)
    return { success = true}
end

---For rerolling the joker/card options
function I.reroll_shop()
    if not G.GAME then
        return{ success = false, error = "Game not avalible" }
    end

    local reroll_cost = G.GAME.current_round.reroll_cost or 5
    if G.GAME.dollars < reroll_cost then
        return { success = false, error = "Cannot afford reroll, cost: " .. tostring(reroll_cost) }
    end

    G.FUNCS.reroll_shop()
    utils.log_input("reroll_shop " .. utils.completed_success_msg)
    return { success = true }
end

---for skipping the shop
function I.skip_shop()
    G.FUNCS.toggle_shop()
    utils.log_input("skip_shop " .. utils.completed_success_msg)
    return { success = true }
end
---for cashing out
function I.cash_out()
    if not G.GAME then
        return { success = false, error = "Game not available" }
    end

    local ok, err = pcall(function()
        G.FUNCS.cash_out({ config = {} })
    end)

    if not ok then
        utils.log_input("cash_out ERROR: " .. tostring(err))
        return { success = false, error = tostring(err) }
    end

    utils.log_input("cash_out " .. utils.completed_success_msg)
    return { success = true }
end

return I