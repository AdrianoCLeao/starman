-- Starman reference gameplay script (M3 gate).
-- Hot-reload safe: only STARMAN_PERSIST is restored across reloads.

STARMAN_PERSIST = STARMAN_PERSIST or { ticks = 0, angle = 0.0 }

starman.log(3, "reference main.lua loaded; persist.ticks=" .. tostring(STARMAN_PERSIST.ticks))

function on_update(dt)
  STARMAN_PERSIST.ticks = (STARMAN_PERSIST.ticks or 0) + 1
  STARMAN_PERSIST.angle = (STARMAN_PERSIST.angle or 0.0) + dt

  -- Broad host API demo: spawn once, then nudge Transform.translation.y
  if STARMAN_PERSIST.entity == nil then
    local handle = starman.spawn("lua_spinner")
    STARMAN_PERSIST.entity = handle
    starman.log(3, "spawned entity handle " .. tostring(handle))
  end

  local handle = STARMAN_PERSIST.entity
  if handle ~= nil then
    local y = math.sin(STARMAN_PERSIST.angle) * 0.5
    -- set_field expects JSON for the leaf value
    local ok, err = pcall(function()
      starman.set_field(handle, "Transform", "translation.y", tostring(y))
    end)
    if not ok then
      starman.log(2, "set_field failed: " .. tostring(err))
    end
  end
end
