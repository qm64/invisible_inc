# Invisible Inc Game Analyser

## Overview

1. The ultimate goal is an ML assisted automated game playing engine for the game Invisible Inc.
1. We will do that in small steps, slowly adding functionality, correcting errors, adding minimal documentation, and keeping a tight structure.

## Game Synopsis

1. The game Invisible Inc is a stealth, turn by turn game.
1. A useful resource is the wiki at https://iiwiki.werp.site/index.
2. The player controls one or more agents, each with different abilities (augments), some with special abilities unique to them.
3. A mission consists of one or more agents infiltrating an enemy facility, to achieve goals, and to escape alive.
4. Different missions have different goals.
5. Some missions have "surprise" goals, which were not known at the start of the mission.
6. There are different campaign modes, ranging from a short campaign of about 5 missions, and extended campaign of perhaps 10 missions, and "endless" mode, which never runs out of missions.
7. Campaigns that end, will end with a special mission. Completing the mission shows the final cutscenes, and game credits.
8. There are many configurable campaign settings, and a few preset campaigns.
9. At the start of most compaigns, the user is allowed to pick 2 agents, and 2 Incognita programs to begin with.
10. Each agent begins with a custom attribute list, and settings.
10. New agents may be picked up on some missions, up to 4 total.
11. Depending on the campaign, 5 or 6 max Incognita program slots are available. (The 6th one is available after a special mission.)
12. Between missions, the user may buy, sell, or redistribute inventory items.
13. Between missions, the user may buy upgrades of the 4 attributes. 

## Special Missions

1. Executive Terminals
- An item with a list of other sites to visit is available. This is also the first mission for every campaign.
1. Vault Key
- A vault key is available, which can be used on other missions to open a vault with extra credits, or a vault with extra augment equipment.
1. Vault
- The Vault with extra credits available, accessible only with a vault key.
1. Final Mission
- Two additional agents join the team, Monst3r and Mother. Each has a special role to play in this final mission.

## Game Engine Approach

1. A game screen capture tool is used to capture screenshots of game play, stored into sessions by timestamp (YYYYMMDD_HHMMSS), in a folder `captures/<session>/frames`. Other subfolders may hold artifacts derived from the frames.
1. A screenshot analyser tool will run across multiple frames and sessions, to determine the areas of the screen where certain elements are always found, by colour, size, and other attributes.
2. A detector framework detects game screen elements. The framework allows pluggable detector elements to be added as they are developed. The detector operates on single frames.
3. A mission analyser gets information from the detector, and adds to the history of the current mission. It allows information to be gleaned from multiple sequential frames if needed. The analyser will assess the current game state, immediate threats and goals, and long term (mission) threats and goals.
4. A mission engine allows control of the game, with the mission analyser used to inform the gameplay engine about the state of the game, etc. (to the full extent of the analyser). The mission engine will also capture screenshots and game state info for later analysis.
5. A campaign analyser/engine will capture the campaign, similar to the mission analyser/engine: campaign state, goals, and choices can be queried; and control the campaign.
6. A game engine will select save slots, campaign settings, agents, Incognita programs, and start and end the campaign. The game engine can run random campaigns, be driven by simple logic (e.g., only sample unique agent / Incognita setups), play campaign settings from a file, or take commands from an external program.
7. An RL trainer will train an RL model to develop good game-playing responses to game states, missions, and campaign scenarios.
8. An RL player will drive the game engine, running missions, campaigns, and full gameplay.

## Coding Guidelines

1. No code is written unless asked.
1. No documentation files are written unless asked.
1. All code should fit together as a whole, except debug scripts. Debug scripts can be one-off.
1. Documentation should be only a few files, tightly coupled, and updated when the corresponding code is updated to the point the documentation is out of date.
1. When appropriate, these tools need to run in parallel. Please add parallel processing, with a default of N-2, where N is the number of available CPUs. Add a command line option to change this number, `--cpus k`, where k is the number of processes to run in parallel.
1. All code should be modular and reusable. All scripts should be possible to import to another script, either for testing, or in a larger framework. 
1. OCR methods should be generic, and take a few parameters such as rectangle, expected colour or masks or method, and allowable character set.

## Game Status Elements

1. Menu hamburger: upper right corner, 3 horizontal lines. Anchor element.
1. Turn info: top edge, towards the right, consisting of the following, separated by a forward slash:
    1. Turn number: "TURN XXX", where XXX is an integer, starting with 1. Assume it can go to 3 digits.
    1. Day number: "DAY YY", where YY is an integer, zero padded.
    1. Campaign mode: Values include TUTORIAL, BEGINNER, EXPERT, EXTENDED, CUSTOM, and others.
    1. Mission facility: Mixed case such as "FTM Executive Terminals". This ends near the menu hamburger.
1. Tactical View polygon: this is at the top edge of the viewport, centred. It says "TACTICAL VIEW" in cyan against green. Anchor element.
    1. Below the tactical view polygon is the wall up/down toggle. This changes the view on the background game display. May be useful to make analysing the game screen easier.
1. Power / Credits, top edge, left corner.
    1. "XX PWR / YYYYYY", in cyan. 
    1. XX is power, 0-20, may be a single digit.
    1. YYYYYY is credits, from 0 to several million (max unknown).
    1. For most of a player's turn, this is visible. It is not shown during some player moves, agent / opponent interactions, or opponent's turn. This is acceptable, as it doesn't change very often -- usually as player actions, or as turn changeover.
1. Incognita rectangle:
    1. This is a wider-than-tall rectangle with a profile of Incognita.
    1. It is often present, similar to power / credits.
    1. Switching to mainframe mode, available Incognita hacking program icons are displayed vertically underneath this. 
    1. Mainframe / normal mode is toggled with <space>, or by clicking this rectangle.
1. Incognita programs:
    1. Program icons are a wider-than-tall rectangle, not as wide as the Incognita rectangle.
    1. Icons are partitioned in two: 
        1. Left side is a number as cost, or turns left, or "dash" for not applicable (e.g., a passive program). If cost, a small "PWR" under the number. If turns, a small "TURNS" under the number.
        1. Right side is a profile image for the program.
    1. Program text: to the right of the program rectangle is text describing what the program does, and possibly cooldown turns to go.
1. Agent Profile Rectangle
    1. Lower left corner, portrait mode.
    1. This is the profile image of the selected agent, drone, or Incognita (in mainframe mode)
    1. The selected agent or drone can be moved or cause actions on the gameplay screen.
    1. In mainframe mode, Incognita's image appears here just to indicate to the player that actions on the gameplay screen will be Incognita's.
    1. The agent profile rectangle may not always present, such as during agent moves, opponent turns, and agent/opponent interactions. However, for most game state analysis, we should generally assume it is present.
1. Agent icons
    1. In a column vertically above the Agent Profile Rectangle, much smaller square icons.
    1. To the right of each icon is an action point value, "XX AP", in cyan, either 1 or 2 digits, from 0-99. The "AP" is all caps, but in a smaller font size than the digits.
1. Agent quick actions:
    1. At the bottom edge, just to the right of the agent profile rectangle, are 4 quick action icons. These are labeled "ACTIONS" in orange, left justified over the icons. 
    1. They are always the same, in the same order: shoot, melee, peek, and sprint.
    1. If they are in colour, they are available. If they are greyed out, they are not available. 
    1. Shoot has an "oversight" mode, and melee has an "ambush" mode, each with an internal decoration.
    1. Peek is a click action.
    1. Sprint pops up a menu above the icon, as a confirmation to sprint or cancel.
1. Augments and inventory
    1. At roughly the midpoint of the agent profile rectangle, to the right, are the augments and inventory icons.
    1. Augments are first, labeled as AUGMENTS in yellow. There are minimally 2 empty square slots, but may be up to 6. Some of these have popup menus for further actions, others only have tool tips for info. Empty slots are difficult to discern from black background. 
    1. To the right of the rightmost augment or empty slot, are the inventory icons, labeled "INVENTORY" in yellow above the icons. 
    1. Up to 8 inventory items are possible.
    1. Inventory items may have popup menus with further actions. They may also have "cooldown" drop-below info rectangles with a number, from 1 to 2 digits.
    1. Some agents start out with no augments. In that case, the augment icons are not present, and neither is the "AUGMENTS" label. "INVENTORY" and the inventory icons start immediately right of the agent profile rectangle.
1. End turn polygon
    1. In the lower right corner is the end turn polygon, "END TURN". Anchor element. Cyan on green
    1. To the left of the end turn polygon, when available, is a rewind polygon, "REWIND". Cyan on green.
    1. Above these 2 polygons are the mission objectives, labeled "OBJECTIVES" in yellow. There is at least 1 row of cyan text, at least 5 is possible (max not known). These objectives change slowly during the mission, as some are achieved, and new ones discovered.

## Viewport

1. The game screen varies according to the resolution and aspect ratio.
1. The "viewport", with game status elements and background game board view, is only slightly larger than the rectangle bounding the anchor elements:
    1. Menu hamburger
    1. Power / Credits
    1. End turn polygon
    1. Agent profile rectangle.
1. While these are not always visible, generally the game screen resolution and aspect ratio doesn't change during the course of a mission.
1. The viewport should be recalculated periodically, perhaps once per minute.
1. The viewport should be verified more often, perhaps every 10s.
1. Viewport verification requires any 3 anchor elements mentioned above to help determine the viewport size and location on the game window.

## Working Method

1. For each fresh chat, start with the baseline code and documentation provided.
1. Some chat sessions may include screenshots or other data generated on previous chats.
1. Take up one task, such as improving the detection of an element, debugging a poor result, or measuring an element placement, etc.
1. Before making changes, give an outline of the changes that will be made, and ask permission to continue or for clarifying questions or details.
1. When progress is made, solidify it by updating the baseline code and documentation, when asked.
1. If progress is made, and code and other artifacts updated, a summary is needed to seed the next chat session.

## What Has Gone Wrong Before

1. Running off into debug circles, without stopping, until the context window is consumed, with no output, artifacts, or additional value added.
1. Generating lots of documentation files about small changes, issues found, etc.
1. Generating one-off scripts to test this or that, cluttering up the local repo folder.
1. Losing progress from previous chats, and making the same mistakes again.
1. Assuming that a detection has found the right element and area, and basing further results on shaky data.

## Game Play

### Vision

1. Guards, facility cameras, drones, and agents are subject to line of sight vision rules.
1. Guards have varying vision rules, including distance, primary vision angle, and peripheral vision angle.
1. Walls and doors block line of sight. 
1. Short objects interfere with line of sight, but not completely. 
1. Vision may "see over" short objects, depending on the distance between the observer and the object.
1. Tall objects cannot be overseen.
1. Agents may "hide" behind objects. There may be a "shadow" close to or immediately adjacent to short objects.

### Items

1. Agents may start with no inventory items, or have no inventory items at various points during the mission.
1. Agents can exchange or give items to other agents, if they are orthogonally (not diagonally).
1. Agents can exchange like items which have a cooldown, to gain tactical advantage. E.g., one agent has positional advantage, or a special ability, and giving that agent a "ready to use" item such as a weapon, cloaking device, or door trap may be an advantageous play.
1. Agents can drop items, which remain in the square they were dropped.
1. Agents can pick up items from the square they were dropped, or adjacent squares.
1. Agents can steal itmes from guards if they are orthogonally adjacent. This may be credits (which go to the credit balance), or inventory items. Inventory items take up an inventory space.
1. Agents can pick up items from downed guards, on the same square, or orthogonally adjacent. Agents can "exchange" items with downed guards. (It is unclear whether the items "given" to a guard remains with the guard, or the square he. was on.)
1. Items can be thrown, either by guards or agents, up to 8 tiles, according to the limits of the game's vision rules (line of sight). "8" is the Euclidean distance, not the Manhattan distance, so roughly a circle of 8 tiles' radius, within line of sight.  (It is unclear if tall objects block thrown objects.)