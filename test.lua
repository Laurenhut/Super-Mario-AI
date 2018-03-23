-- Compatibility of the memory read/write functions
local u8 =  memory.readbyte
local s8 =  memory.readsbyte
local w8 =  memory.writebyte
local u16 = memory.readword
local s16 = memory.readsword
local w16 = memory.writeword
local u24 = memory.readhword
local s24 = memory.readshword
local w24 = memory.writehword
local u32 = memory.readdword
local s32 = memory.readsdword
local w32 = memory.writedword



local Sprites_info = {}  -- keeps track of useful sprite info that might be used outside the main sprite function
local Sprite_hitbox = {}  -- keeps track of what sprite slots must display the hitbox
-- local luap = require "luap"
-- local config = require "config"
-- local COLOUR = config.COLOUR

require"zmq"
 a=0;
 send={}
 i=0;

local smw = {}
 smw.constant = {
   -- Game Modes
   game_mode_overworld = 0x0e,
   game_mode_fade_to_level = 0x0f,
   game_mode_level = 0x14,

   -- Sprites
   sprite_max = 12,
   extended_sprite_max = 10,
   cluster_sprite_max = 20,
   minor_extended_sprite_max = 12,
   bounce_sprite_max = 4,
   null_sprite_id = 0xff,

   -- Blocks
   blank_tile_map16 = 0x25,
 }

 smw.WRAM = {
   -- I/O
   ctrl_1_1 = 0x0015,
   ctrl_1_2 = 0x0017,
   firstctrl_1_1 = 0x0016,
   firstctrl_1_2 = 0x0018,

   -- General
   game_mode = 0x0100,
   real_frame = 0x0013,
   effective_frame = 0x0014,
   lag_indicator = 0x01fe,
   timer_frame_counter = 0x0f30,
   RNG = 0x148d,
   current_level = 0x00fe,  -- plus 1
   sprite_memory_header = 0x1692,
   lock_animation_flag = 0x009d, -- Most codes will still run if this is set, but almost nothing will move or animate.
   level_mode_settings = 0x1925,
   star_road_speed = 0x1df7,
   star_road_timer = 0x1df8,
   current_character = 0x0db3, -- #00 = Mario, #01 = Luigi
   exit_counter = 0x1F2E,
   event_flags = 0x1F02, -- 15 bytes (1 bit per exit)
   timer = 0x0F31, -- 3 bytes, one for each digit

   -- Cheats
   frozen = 0x13fb,
   level_paused = 0x13d4,
   level_index = 0x13bf,
   room_index = 0x00ce,
   level_flag_table = 0x1ea2,
   level_exit_type = 0x0dd5,
   midway_point = 0x13ce,

   -- Camera
   layer1_x_mirror = 0x1a,
   layer1_y_mirror = 0x1c,
   layer1_VRAM_left_up = 0x4d,
   layer1_VRAM_right_down = 0x4f,
   camera_x = 0x1462,
   camera_y = 0x1464,
   camera_left_limit = 0x142c,
   camera_right_limit = 0x142e,
   screens_number = 0x005d,
   hscreen_number = 0x005e,
   vscreen_number = 0x005f,
   vertical_scroll_flag_header = 0x1412,  -- #$00 = Disable; #$01 = Enable; #$02 = Enable if flying/climbing/etc.
   vertical_scroll_enabled = 0x13f1,
   camera_scroll_timer = 0x1401,

   -- Sprites
   sprite_status = 0x14c8,
   sprite_number = 0x009e,
   sprite_x_high = 0x14e0,
   sprite_x_low = 0x00e4,
   sprite_y_high = 0x14d4,
   sprite_y_low = 0x00d8,
   sprite_x_sub = 0x14f8,
   sprite_y_sub = 0x14ec,
   sprite_x_speed = 0x00b6,
   sprite_y_speed = 0x00aa,
   sprite_x_offscreen = 0x15a0,
   sprite_y_offscreen = 0x186c,
   sprite_OAM_xoff = 0x0304,
   sprite_OAM_yoff = 0x0305,
   sprite_being_eaten_flag = 0x15d0,
   sprite_OAM_index = 0x15ea,
   sprite_swap_slot = 0x1861,
   sprite_miscellaneous1 = 0x00c2,
   sprite_miscellaneous2 = 0x1504,
   sprite_miscellaneous3 = 0x1510,
   sprite_miscellaneous4 = 0x151c,
   sprite_miscellaneous5 = 0x1528,
   sprite_miscellaneous6 = 0x1534,
   sprite_miscellaneous7 = 0x1540,
   sprite_miscellaneous8 = 0x154c,
   sprite_miscellaneous9 = 0x1558,
   sprite_miscellaneous10 = 0x1564,
   sprite_miscellaneous11 = 0x1570,
   sprite_miscellaneous12 = 0x157c,
   sprite_miscellaneous13 = 0x1594,
   sprite_miscellaneous14 = 0x15ac,
   sprite_miscellaneous15 = 0x1602,
   sprite_miscellaneous16 = 0x160e,
   sprite_miscellaneous17 = 0x1626,
   sprite_miscellaneous18 = 0x163e,
   sprite_miscellaneous19 = 0x187b,
   sprite_underwater = 0x164a,
   sprite_disable_cape = 0x1fe2,
   sprite_1_tweaker = 0x1656,
   sprite_2_tweaker = 0x1662,
   sprite_3_tweaker = 0x166e,
   sprite_4_tweaker = 0x167a,
   sprite_5_tweaker = 0x1686,
   sprite_6_tweaker = 0x190f,
   sprite_tongue_wait = 0x14a3,
   sprite_yoshi_squatting = 0x18af,
   sprite_buoyancy = 0x190e,
   sprite_index_to_level = 0x161A,
   sprite_data_pointer = 0x00CE, -- 3 bytes
   sprite_load_status_table = 0x1938, -- 128 bytes
   bowser_attack_timers = 0x14b0, -- 9 bytes

   -- Extended sprites
   extspr_number = 0x170b,
   extspr_x_high = 0x1733,
   extspr_x_low = 0x171f,
   extspr_y_high = 0x1729,
   extspr_y_low = 0x1715,
   extspr_x_speed = 0x1747,
   extspr_y_speed = 0x173d,
   extspr_suby = 0x1751,
   extspr_subx = 0x175b,
   extspr_table = 0x1765,
   extspr_table2 = 0x176f,

   -- Cluster sprites
   cluspr_flag = 0x18b8,
   cluspr_number = 0x1892,
   cluspr_x_high = 0x1e3e,
   cluspr_x_low = 0x1e16,
   cluspr_y_high = 0x1e2a,
   cluspr_y_low = 0x1e02,
   cluspr_timer = 0x0f9a,
   cluspr_table_1 = 0x0f4a,
   cluspr_table_2 = 0x0f72,
   cluspr_table_3 = 0x0f86,
   reappearing_boo_counter = 0x190a,

   -- Minor extended sprites
   minorspr_number = 0x17f0,
   minorspr_x_high = 0x18ea,
   minorspr_x_low = 0x1808,
   minorspr_y_high = 0x1814,
   minorspr_y_low = 0x17fc,
   minorspr_xspeed = 0x182c,
   minorspr_yspeed = 0x1820,
   minorspr_x_sub = 0x1844,
   minorspr_y_sub = 0x1838,
   minorspr_timer = 0x1850,

   -- Bounce sprites
   bouncespr_number = 0x1699,
   bouncespr_x_high = 0x16ad,
   bouncespr_x_low = 0x16a5,
   bouncespr_y_high = 0x16a9,
   bouncespr_y_low = 0x16a1,
   bouncespr_timer = 0x16c5,
   bouncespr_last_id = 0x18cd,
   turn_block_timer = 0x18ce,

   -- Player
   x = 0x0094,
   y = 0x0096,
   previous_x = 0x00d1,
   previous_y = 0x00d3,
   x_sub = 0x13da,
   y_sub = 0x13dc,
   x_speed = 0x007b,
   x_subspeed = 0x007a,
   y_speed = 0x007d,
   direction = 0x0076,
   is_ducking = 0x0073,
   p_meter = 0x13e4,
   take_off = 0x149f,
   powerup = 0x0019,
   cape_spin = 0x14a6,
   cape_fall = 0x14a5,
   cape_interaction = 0x13e8,
   flight_animation = 0x1407,
   diving_status = 0x1409,
   player_animation_trigger = 0x0071,
   climbing_status = 0x0074,
   spinjump_flag = 0x140d,
   player_blocked_status = 0x0077,
   item_box = 0x0dc2,
   cape_x = 0x13e9,
   cape_y = 0x13eb,
   on_ground = 0x13ef,
   on_ground_delay = 0x008d,
   on_air = 0x0072,
   can_jump_from_water = 0x13fa,
   carrying_item = 0x148f,
   player_pose_turning = 0x1499,
   mario_score = 0x0f34,
   player_coin = 0x0dbf,
   player_looking_up = 0x13de,
   OW_x = 0x1f17,
   OW_y = 0x1f19,

   -- Yoshi
   yoshi_riding_flag = 0x187a,  -- #$00 = No, #$01 = Yes, #$02 = Yes, and turning around.
   yoshi_tile_pos = 0x0d8c,
   yoshi_in_pipe = 0x1419,

   -- Timer
   --keep_mode_active = 0x0db1,
   pipe_entrance_timer = 0x0088,
   score_incrementing = 0x13d6,
   fadeout_radius = 0x1433,
   peace_image_timer = 0x1492,
   end_level_timer = 0x1493,
   multicoin_block_timer = 0x186b,
   gray_pow_timer = 0x14ae,
   blue_pow_timer = 0x14ad,
   dircoin_timer = 0x190c,
   pballoon_timer = 0x1891,
   star_timer = 0x1490,
   animation_timer = 0x1496,
   invisibility_timer = 0x1497,
   fireflower_timer = 0x149b,
   yoshi_timer = 0x18e8,
   swallow_timer = 0x18ac,
   lakitu_timer = 0x18e0,
   spinjump_fireball_timer = 0x13e2,
   game_intro_timer = 0x1df5,
   pause_timer = 0x13d3,
   bonus_timer = 0x14ab,
   disappearing_sprites_timer = 0x18bf,
   message_box_timer = 0x1b89,

   -- Layers
   layer2_x_nextframe = 0x1466,
   layer2_y_nextframe = 0x1468,
 }

 smw.HITBOX_SPRITE = {  -- sprites' hitbox against player and other sprites
   [0x00] = { xoff = 2, yoff = 3, width = 12, height = 10, oscillation = true },
   [0x01] = { xoff = 2, yoff = 3, width = 12, height = 21, oscillation = true },
   [0x02] = { xoff = 16, yoff = -2, width = 16, height = 18, oscillation = true },
   [0x03] = { xoff = 20, yoff = 8, width = 8, height = 8, oscillation = true },
   [0x04] = { xoff = 0, yoff = -2, width = 48, height = 14, oscillation = true },
   [0x05] = { xoff = 0, yoff = -2, width = 80, height = 14, oscillation = true },
   [0x06] = { xoff = 1, yoff = 2, width = 14, height = 24, oscillation = true },
   [0x07] = { xoff = 8, yoff = 8, width = 40, height = 48, oscillation = true },
   [0x08] = { xoff = -8, yoff = -2, width = 32, height = 16, oscillation = true },
   [0x09] = { xoff = -2, yoff = 8, width = 20, height = 30, oscillation = true },
   [0x0a] = { xoff = 3, yoff = 7, width = 1, height = 2, oscillation = true },
   [0x0b] = { xoff = 6, yoff = 6, width = 3, height = 3, oscillation = true },
   [0x0c] = { xoff = 1, yoff = -2, width = 13, height = 22, oscillation = true },
   [0x0d] = { xoff = 0, yoff = -4, width = 15, height = 16, oscillation = true },
   [0x0e] = { xoff = 6, yoff = 6, width = 20, height = 20, oscillation = true },
   [0x0f] = { xoff = 2, yoff = -2, width = 36, height = 18, oscillation = true },
   [0x10] = { xoff = 0, yoff = -2, width = 15, height = 32, oscillation = true },
   [0x11] = { xoff = -24, yoff = -24, width = 64, height = 64, oscillation = true },
   [0x12] = { xoff = -4, yoff = 16, width = 8, height = 52, oscillation = true },
   [0x13] = { xoff = -4, yoff = 16, width = 8, height = 116, oscillation = true },
   [0x14] = { xoff = 4, yoff = 2, width = 24, height = 12, oscillation = true },
   [0x15] = { xoff = 0, yoff = -2, width = 15, height = 14, oscillation = true },
   [0x16] = { xoff = -4, yoff = -12, width = 24, height = 24, oscillation = true },
   [0x17] = { xoff = 2, yoff = 8, width = 12, height = 69, oscillation = true },
   [0x18] = { xoff = 2, yoff = 19, width = 12, height = 58, oscillation = true },
   [0x19] = { xoff = 2, yoff = 35, width = 12, height = 42, oscillation = true },
   [0x1a] = { xoff = 2, yoff = 51, width = 12, height = 26, oscillation = true },
   [0x1b] = { xoff = 2, yoff = 67, width = 12, height = 10, oscillation = true },
   [0x1c] = { xoff = 0, yoff = 10, width = 10, height = 48, oscillation = true },
   [0x1d] = { xoff = 2, yoff = -3, width = 28, height = 27, oscillation = true },
   [0x1e] = { xoff = 6, yoff = -8, width = 3, height = 32, oscillation = true },  -- default: { xoff = -32, yoff = -8, width = 48, height = 32, oscillation = true },
   [0x1f] = { xoff = -16, yoff = -4, width = 48, height = 18, oscillation = true },
   [0x20] = { xoff = -4, yoff = -24, width = 8, height = 24, oscillation = true },
   [0x21] = { xoff = -4, yoff = 16, width = 8, height = 24, oscillation = true },
   [0x22] = { xoff = 0, yoff = 0, width = 16, height = 16, oscillation = true },
   [0x23] = { xoff = -8, yoff = -24, width = 32, height = 32, oscillation = true },
   [0x24] = { xoff = -12, yoff = 32, width = 56, height = 56, oscillation = true },
   [0x25] = { xoff = -14, yoff = 4, width = 60, height = 20, oscillation = true },
   [0x26] = { xoff = 0, yoff = 88, width = 32, height = 8, oscillation = true },
   [0x27] = { xoff = -4, yoff = -4, width = 24, height = 24, oscillation = true },
   [0x28] = { xoff = -14, yoff = -24, width = 28, height = 40, oscillation = true },
   [0x29] = { xoff = -16, yoff = -4, width = 32, height = 27, oscillation = true },
   [0x2a] = { xoff = 2, yoff = -8, width = 12, height = 19, oscillation = true },
   [0x2b] = { xoff = 0, yoff = 2, width = 16, height = 76, oscillation = true },
   [0x2c] = { xoff = -8, yoff = -8, width = 16, height = 16, oscillation = true },
   [0x2d] = { xoff = 4, yoff = 4, width = 8, height = 4, oscillation = true },
   [0x2e] = { xoff = 2, yoff = -2, width = 28, height = 34, oscillation = true },
   [0x2f] = { xoff = 2, yoff = -2, width = 28, height = 32, oscillation = true },
   [0x30] = { xoff = 8, yoff = -14, width = 16, height = 28, oscillation = true },
   [0x31] = { xoff = 0, yoff = -2, width = 48, height = 18, oscillation = true },
   [0x32] = { xoff = 0, yoff = -2, width = 48, height = 18, oscillation = true },
   [0x33] = { xoff = 0, yoff = -2, width = 64, height = 18, oscillation = true },
   [0x34] = { xoff = -4, yoff = -4, width = 8, height = 8, oscillation = true },
   [0x35] = { xoff = 3, yoff = 0, width = 18, height = 32, oscillation = true },
   [0x36] = { xoff = 8, yoff = 8, width = 52, height = 46, oscillation = true },
   [0x37] = { xoff = 0, yoff = -8, width = 15, height = 20, oscillation = true },
   [0x38] = { xoff = 8, yoff = 16, width = 32, height = 40, oscillation = true },
   [0x39] = { xoff = 4, yoff = 3, width = 8, height = 10, oscillation = true },
   [0x3a] = { xoff = -8, yoff = 16, width = 32, height = 16, oscillation = true },
   [0x3b] = { xoff = 0, yoff = 0, width = 16, height = 13, oscillation = true },
   [0x3c] = { xoff = 12, yoff = 10, width = 3, height = 6, oscillation = true },
   [0x3d] = { xoff = 12, yoff = 21, width = 3, height = 20, oscillation = true },
   [0x3e] = { xoff = 16, yoff = 18, width = 254, height = 16, oscillation = true },
   [0x3f] = { xoff = 8, yoff = 8, width = 8, height = 24, oscillation = true }
 }

 smw.OBJ_CLIPPING_SPRITE = {  -- sprites' interaction points against objects
   [0x0] = {xright = 14, xleft =  2, xdown =  8, xup =  8, yright =  8, yleft =  8, ydown = 16, yup =  2},
   [0x1] = {xright = 14, xleft =  2, xdown =  7, xup =  7, yright = 18, yleft = 18, ydown = 32, yup =  2},
   [0x2] = {xright =  7, xleft =  7, xdown =  7, xup =  7, yright =  7, yleft =  7, ydown =  7, yup =  7},
   [0x3] = {xright = 14, xleft =  2, xdown =  8, xup =  8, yright = 16, yleft = 16, ydown = 32, yup = 11},
   [0x4] = {xright = 16, xleft =  0, xdown =  8, xup =  8, yright = 18, yleft = 18, ydown = 32, yup =  2},
   [0x5] = {xright = 13, xleft =  2, xdown =  8, xup =  8, yright = 24, yleft = 24, ydown = 32, yup = 16},
   [0x6] = {xright =  7, xleft =  0, xdown =  4, xup =  4, yright =  4, yleft =  4, ydown =  8, yup =  0},
   [0x7] = {xright = 31, xleft =  1, xdown = 16, xup = 16, yright = 16, yleft = 16, ydown = 31, yup =  1},
   [0x8] = {xright = 15, xleft =  0, xdown =  8, xup =  8, yright =  8, yleft =  8, ydown = 15, yup =  0},
   [0x9] = {xright = 16, xleft =  0, xdown =  8, xup =  8, yright =  8, yleft =  8, ydown = 16, yup =  0},
   [0xa] = {xright = 13, xleft =  2, xdown =  8, xup =  8, yright = 72, yleft = 72, ydown = 80, yup = 66},
   [0xb] = {xright = 14, xleft =  2, xdown =  8, xup =  8, yright =  4, yleft =  4, ydown =  8, yup =  0},
   [0xc] = {xright = 13, xleft =  2, xdown =  8, xup =  8, yright =  0, yleft =  0, ydown =  0, yup =  0},
   [0xd] = {xright = 16, xleft =  0, xdown =  8, xup =  8, yright =  8, yleft =  8, ydown = 16, yup =  0},
   [0xe] = {xright = 31, xleft =  0, xdown = 16, xup = 16, yright =  8, yleft =  8, ydown = 16, yup =  0},
   [0xf] = {xright =  8, xleft =  8, xdown =  8, xup = 16, yright =  4, yleft =  1, ydown =  2, yup =  4}
 }

local SMW = smw.constant
local OBJ_CLIPPING_SPRITE = smw.OBJ_CLIPPING_SPRITE
local WRAM = smw.WRAM
local HITBOX_SPRITE = smw.HITBOX_SPRITE

-- In level frequently used info
Player_animation_trigger = u8("WRAM", WRAM.player_animation_trigger)
Player_powerup = u8("WRAM", WRAM.powerup)
Camera_x = s16("WRAM", WRAM.camera_x)
Camera_y = s16("WRAM", WRAM.camera_y)
Yoshi_riding_flag = u8("WRAM", WRAM.yoshi_riding_flag) ~= 0
Player_x = s16("WRAM", WRAM.x)
Player_y = s16("WRAM", WRAM.y)



-- marios x and y coordinates

previous_x = 0x00d1
previous_y = 0x00d3
on_ground = 0x13ef
player_blocked_status = 0x0077
sprite_number = 0x14e0
sprite_x_speed = 0x00b6
end_level_timer = 0x1493
mario_score = 0x0f34
sprite_swap_slot = 0x1861
sprite_memory_header = 0x1692


for i = 0, SMW.sprite_max -1 do
  Sprites_info[i] = {}
end
for key = 0, SMW.sprite_max - 1 do
  Sprite_hitbox[key] = {}
  -- for number = 0, 0xff do
  --   Sprite_hitbox[key][number] = {["sprite"] = true, ["block"] = GOOD_SPRITES_CLIPPING[number]}
  -- end
end



-- unsigned to signed (based in <bits> bits)
function signed16(num)
  local maxval = 32768
  if num < maxval then return num else return num - 2*maxval end
end

-- Converts the in-game (x, y) to SNES-screen coordinates
local function screen_coordinates(x, y, camera_x, camera_y)
  local x_screen = (x - camera_x)
  local y_screen = (y - camera_y)

  return x_screen, y_screen
end

Player_x_screen, Player_y_screen = screen_coordinates(Player_x, Player_y, Camera_x, Camera_y)
-- Display.is_player_near_borders = Player_x_screen <= 32 or Player_x_screen >= 0xd0 or Player_y_screen <= -100 or Player_y_screen >= 224


 local function sprite_info(id, counter)
   local t = Sprites_info[id]
   local sprite_status = t.status
   if sprite_status == 0 then return 0 end -- returns if the slot is empty

   local x = t.x
   local y = t.y
   local x_sub = t.x_sub
   local y_sub = t.y_sub
   local number = t.number
   local stun = t.stun
   local x_speed = t.x_speed
   local y_speed = t.y_speed
   local contact_mario = t.contact_mario
   local underwater = t.underwater
   local x_offscreen = t.x_offscreen
   local y_offscreen = t.y_offscreen
   local x_screen = t.x_screen
   local y_screen = t.y_screen
   local xpt_left = t.xpt_left
   local xpt_right = t.xpt_right
   local ypt_up = t.ypt_up
   local ypt_down = t.ypt_down
   local xoff = t.hitbox_xoff
   local yoff = t.hitbox_yoff
   local sprite_width = t.hitbox_width
   local sprite_height = t.hitbox_height

   -- HUD elements
   local oscillation_flag = t.oscillation_flag
   local info_color = t.info_color
   local color_background = t.background_color

   -- draw_sprite_hitbox(id)

   -- Special sprites analysis:
   -- local fn = special_sprite_property[number]
   -- if fn then fn(id) end

   -- Print those informations next to the sprite
   -- draw.Font = "Uzebox6x8"
   -- draw.Text_opacity = 1.0
   -- draw.Bg_opacity = 1.0
   --
   -- if x_offscreen ~= 0 or y_offscreen ~= 0 then
   --   draw.Text_opacity = 0.6
   -- end

   local contact_str = contact_mario == 0 and "" or " " .. contact_mario

   local sprite_middle = t.sprite_middle
   local sprite_top = t.sprite_top

    --draw.text(draw.AR_x*sprite_middle, draw.AR_y*sprite_top, fmt("#%.2d%s", id, contact_str), info_color, true, false, 0.5, 1.0)

    --id
    -- gui.text(15,12, id)
    -- --contact str
    -- gui.text(35,12,contact_str)


    -- if Player_powerup == 2 then
    --   local contact_cape = u8("WRAM", WRAM.sprite_disable_cape + id)
    --   if contact_cape ~= 0 then
    --     draw.text(draw.AR_x*sprite_middle, draw.AR_y*sprite_top - 2*draw.font_height(), contact_cape, COLOUR.cape, true)
    --   end
    -- end


   -- Sprite tweakers info
   -- sprite_tweaker_editor(id)

   -- The sprite table:

    -- draw.Font = false
    local x_speed_water = ""
    if underwater ~= 0 then  -- if sprite is underwater
      local correction = 3 * math.floor(math.floor(x_speed/2) / 2)
      x_speed_water = string.format("%+.2d=%+.2d", correction - x_speed, correction)
    end
    local sprite_str = "#%02d %02x %s%d.%1x(%+.2d%s) %d.%1x(%+.2d)",
       id, number, t.table_special_info, x, math.floor(x_sub/16), x_speed, x_speed_water, y, math.floor(y_sub/16), y_speed

    --gui.text(draw.Buffer_width + draw.Border_right, table_position + counter*draw.font_height(), sprite_str, info_color, true)
    -- gui.text(15,32,id)
    -- gui.text(25,32,number)
    -- gui.text(35,32,t.table_special_info)
    -- gui.text(45,32,x)
    -- -- gui.text(55,32, math.floor(x_sub/16))
    -- gui.text(85,32,x_speed)
    -- -- -- gui.text(75,32,x_speed_water)
    -- gui.text(105,32,y)
    -- -- gui.text(95,32,math.floor(y_sub/16))
    -- gui.text(145,32, y_speed)



   -- return 1
   return 1, id, contact_str, x, x_speed,y,y_speed
 end

 local function scan_sprite_info(lua_table, slot)
   local t = lua_table[slot]
   if not t then error"Wrong Sprite table" end

   t.status = u8("WRAM", WRAM.sprite_status + slot)
   if t.status == 0 then
     return -- returns if the slot is empty
   end

   local x = 256*u8("WRAM", WRAM.sprite_x_high + slot) + u8("WRAM", WRAM.sprite_x_low + slot)
   local y = 256*u8("WRAM", WRAM.sprite_y_high + slot) + u8("WRAM", WRAM.sprite_y_low + slot)
   t.x_sub = u8("WRAM", WRAM.sprite_x_sub + slot)
   t.y_sub = u8("WRAM", WRAM.sprite_y_sub + slot)
   t.number = u8("WRAM", WRAM.sprite_number + slot)
   t.stun = u8("WRAM", WRAM.sprite_miscellaneous7 + slot)
   t.x_speed = s8("WRAM", WRAM.sprite_x_speed + slot)
   t.y_speed = s8("WRAM", WRAM.sprite_y_speed + slot)
   t.contact_mario = u8("WRAM", WRAM.sprite_miscellaneous8 + slot)
   t.underwater = u8("WRAM", WRAM.sprite_underwater + slot)
   t.x_offscreen = s8("WRAM", WRAM.sprite_x_offscreen + slot)
   t.y_offscreen = s8("WRAM", WRAM.sprite_y_offscreen + slot)

   -- Transform some read values into intelligible content
   t.x = signed16(x)
   t.y = signed16(y)
   t.x_screen, t.y_screen = screen_coordinates(t.x, t.y, Camera_x, Camera_y)

   -- if OPTIONS.display_debug_sprite_extra or ((t.status < 0x8 and t.status > 0xb) or stun ~= 0) then
   --   t.table_special_info = fmt("(%d %d) ", t.status, t.stun)
   -- else
   --   t.table_special_info = ""
   -- end

   -- t.oscillation_flag = bit.test(u8("WRAM", WRAM.sprite_4_tweaker + slot), 5) or OSCILLATION_SPRITES[t.number]

   -- Sprite clipping vs mario and sprites
   local boxid = bit.band(u8("WRAM", WRAM.sprite_2_tweaker + slot), 0x3f)  -- This is the type of box of the sprite
   t.hitbox_id = boxid
   t.hitbox_xoff = HITBOX_SPRITE[boxid].xoff
   t.hitbox_yoff = HITBOX_SPRITE[boxid].yoff
   t.hitbox_width = HITBOX_SPRITE[boxid].width
   t.hitbox_height = HITBOX_SPRITE[boxid].height

   -- Sprite clipping vs objects
   local clip_obj = bit.band(u8("WRAM", WRAM.sprite_1_tweaker + slot), 0xf)  -- type of hitbox for blocks
   t.clipping_id = clip_obj
   t.xpt_right = OBJ_CLIPPING_SPRITE[clip_obj].xright
   t.ypt_right = OBJ_CLIPPING_SPRITE[clip_obj].yright
   t.xpt_left = OBJ_CLIPPING_SPRITE[clip_obj].xleft
   t.ypt_left = OBJ_CLIPPING_SPRITE[clip_obj].yleft
   t.xpt_down = OBJ_CLIPPING_SPRITE[clip_obj].xdown
   t.ypt_down = OBJ_CLIPPING_SPRITE[clip_obj].ydown
   t.xpt_up = OBJ_CLIPPING_SPRITE[clip_obj].xup
   t.ypt_up = OBJ_CLIPPING_SPRITE[clip_obj].yup


   t.sprite_middle = t.x_screen + t.hitbox_xoff + math.floor(t.hitbox_width/2)
   t.sprite_top = t.y_screen + math.min(t.hitbox_yoff, t.ypt_up)
 end


 local function sprites()
   local counter = 0
   id1={} --sprite id
   contact_str={} --has mario contacted sprite
   x={}-- sprite x pos
   x_speed={} -- sprite x speed
   y={} -- sprite y pos
   y_speed={} -- sprite y speed

   -- local table_position = draw.AR_y*40 -- lsnes
   for id = 0, SMW.sprite_max - 1 do
     scan_sprite_info(Sprites_info, id)

     th, id1[id], contact_str[id], x[id], x_speed[id],y[id],y_speed[id]= sprite_info(id, counter)
     counter = counter + th
   end


    local swap_slot = u8("WRAM", WRAM.sprite_swap_slot)
    local smh = u8("WRAM", WRAM.sprite_memory_header)
    -- USED draw.text(draw.Buffer_width + draw.Border_right, table_position - 2*draw.font_height(), fmt("spr:%.2d ", counter), COLOUR.weak, true)
    -- NOT USED  draw.text(draw.Buffer_width + draw.Border_right, table_position - draw.font_height(), fmt("1st div: %d. Swap: %d ",
    --                                        SPRITE_MEMORY_MAX[smh] or 0, swap_slot), COLOUR.weak, true)
    --#sprites
    -- gui.text(15,60,"spr: ")
    -- gui.text(65,60, counter)

    return counter, id1, contact_str, x, x_speed,y,y_speed


 end

function on_paint()
    counter, id1, contact_str, x, x_speed,y,y_speed=sprites()
    --counter # sprites on screen
    -- th=id1[9]
    -- th1=contact_str[9]
    -- th2=x[9]
    -- th3=x_speed[9]
    -- th4=y[9]
    -- th5=y_speed[9]
    -- th6=counter
    -- --id
    -- gui.text(15,12, tostring(th))
    -- --contact str
    -- gui.text(35,12,tostring(th1))
    -- gui.text(45,32,tostring(th2))
    -- -- gui.text(55,32, math.floor(x_sub/16))
    -- gui.text(85,32,tostring(th3))
    -- -- -- gui.text(75,32,x_speed_water)
    -- gui.text(105,32,tostring(th4))
    -- -- gui.text(95,32,math.floor(y_sub/16))
    -- gui.text(145,32, tostring(th5))
    -- --#sprites
    -- gui.text(15,60,"# spr: ")
    -- gui.text(65,60, tostring(th6))


    if a==50 then
        Player_x = s16("WRAM", 0x0094)
        Player_y = s16("WRAM", 0x0096)
        -- gui.text(15,12,Player_x) -- will give pixel position
        Player_e = s16("WRAM", end_level_timer)
        Player_s = s16("WRAM", mario_score)
        Player_g = s16("WRAM",on_ground)
        send[0]=Player_x -- will give pixel position x
        send[1]= Player_e -- will become not zero when reach end
        send[2]=Player_s -- current score
        send[3]=Player_g -- if mario on ground 1=y 0=n
        send[4]=Player_y
        -- gui.text(15,12,send[0]) -- will give pixel position
        -- gui.text(15,32,send[1]) -- will become not zero when reach end
        -- gui.text(15,62,send[2])

        local context = zmq.init(1)
        local socket = context:socket(zmq.REQ)
        socket:connect("tcp://localhost:5555")
        -- while  i<3 do
        --     socket:send(send[i])
        --     i=i+1
        -- end
        -- i=0
        socket:send(send[0],zmq.SNDMORE )
        socket:send(send[4],zmq.SNDMORE )
        socket:send(send[1],zmq.SNDMORE )
        socket:send(send[2],zmq.SNDMORE )
        socket:send(send[3],zmq.SNDMORE )



        for j = 0, SMW.sprite_max - 1 do
            -- socket:send(tostring(id1[j]),zmq.SNDMORE )

            if id1[j] == nil then
                socket:send(tostring(-1),zmq.SNDMORE )
            else
                socket:send(tostring(id1[j]),zmq.SNDMORE )
            end



            if contact_str[j] == nil or contact_str[j] == ''  then
                socket:send(tostring(0),zmq.SNDMORE )
            else
                socket:send(tostring(contact_str[j]),zmq.SNDMORE )
            end


            if x[j] == nil then
                socket:send(tostring(-1),zmq.SNDMORE )
            else
                socket:send(tostring(x[j]),zmq.SNDMORE )
            end

            -- socket:send(tostring(x[j]),zmq.SNDMORE )


            -- socket:send(tostring(x_speed[j]),zmq.SNDMORE )
            if x_speed[j] == nil then
                socket:send(tostring(0),zmq.SNDMORE )
            else
                socket:send(tostring(x_speed[j]),zmq.SNDMORE )
            end

            if y[j] == nil then
                socket:send(tostring(-1),zmq.SNDMORE )
            else
                socket:send(tostring(y[j]),zmq.SNDMORE )
            end
            -- socket:send(tostring(y[j]),zmq.SNDMORE )

            if y_speed[j] == nil then
                socket:send(tostring(0),zmq.SNDMORE )
            else
                socket:send(tostring(y_speed[j]),zmq.SNDMORE )
            end
            -- socket:send(tostring(y_speed[j]),zmq.SNDMORE )
        end
        -- socket:send(counter)
        if counter == nil then
            socket:send(tostring(0) )
        else
            socket:send(counter )
        end
        -- socket:send(send[3])
        -- socket:send(send[0] )
        -- socket:send(send[2])
        -- local reply = socket:recv()
        socket:close()
        context:term()

        gui.text(15,12,a) -- will give pixel position
        a=0
    else
        gui.text(15,12,a)
        a=a+1

    end





    -- Player_x = s16("WRAM", 0x0094)
    --gui.text(15,12,Player_x) -- will give pixel position
    -- Player_s = s16("WRAM",sprite_number)
    -- sprite_number = u8("WRAM", 0x16cd + id)
    -- Player_block = s16("WRAM", player_blocked_status)

   -- gui.text(30,12,Player_ground) -- 1=yes 0=no
   -- gui.text(15,30,Player_s) --?

end
gui.repaint() -- Tell lsnes to update display when we run this script!
