--[[
sequential-reveal.lua: keep slides in reading order (reveal.js only).

With `incremental: true`, list items appear one at a time, but anything after
the list (a table, a closing sentence, a formula) would be visible from the
start. This filter turns every such block into a fragment, so it appears only
after the list, in source order.

- Figures and raw HTML (iframes with animations) are left visible: they are
  the context the speaker talks over.
- Opt out for one block with `::: {.static}` (or `::: {.nonincremental}`).
- `::: {.incremental-rows}` around a table reveals it row by row.
- A slide boundary is any header of level 1 to 3.
]]

local CONTAINERS = { columns = true, column = true, ["smaller-text"] = true,
                     ["smallest-text"] = true, ["incremental-rows"] = true }
local KEEP = { fragment = true, aside = true, notes = true, static = true,
               nonincremental = true }

local function has_class(el, names)
  if not el.classes then return false end
  for _, c in ipairs(el.classes) do
    if names[c] or c:match("^content%-") then return true end
  end
  return false
end

local function is_container(el)
  if el.t ~= "Div" then return false end
  for _, c in ipairs(el.classes) do
    if CONTAINERS[c] then return true end
  end
  return false
end

local function is_figure(b)
  return b.t == "Figure"
      or (b.t == "Para" and #b.content == 1 and b.content[1].t == "Image")
end

-- Does this block contain a list that will be revealed item by item?
local function has_incremental_list(b)
  if b.t == "BulletList" or b.t == "OrderedList" then return true end
  if b.t == "Div" then
    if b.classes:includes("nonincremental") then return false end
    for _, c in ipairs(b.content) do
      if has_incremental_list(c) then return true end
    end
  end
  return false
end

local function reveal_rows(div)
  return div:walk({
    Table = function(t)
      for _, body in ipairs(t.bodies) do
        for _, row in ipairs(body.body) do
          row.attr = pandoc.Attr("", { "fragment" })
        end
      end
      return t
    end,
  })
end

local function process(blocks, seen)
  local out = pandoc.List()
  for _, b in ipairs(blocks) do
    if b.t == "Header" and b.level <= 3 then
      seen = false
      out:insert(b)
    elseif b.t == "Div" and has_class(b, KEEP) then
      out:insert(b)
    elseif is_container(b) then
      if b.classes:includes("incremental-rows") then b = reveal_rows(b) end
      local inner
      inner, seen = process(b.content, seen)
      b.content = inner
      out:insert(b)
    elseif has_incremental_list(b) then
      seen = true
      out:insert(b)
    elseif seen and not is_figure(b) and b.t ~= "RawBlock" then
      out:insert(pandoc.Div({ b }, pandoc.Attr("", { "fragment" })))
    else
      out:insert(b)
    end
  end
  return out, seen
end

function Pandoc(doc)
  if not quarto.doc.is_format("revealjs") then return nil end
  doc.blocks = process(doc.blocks, false)
  return doc
end
