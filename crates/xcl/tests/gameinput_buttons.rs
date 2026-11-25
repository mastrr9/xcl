//! The controller's buttons as a title reads them (`hle::gameinput`): XInput's button word turned
//! into GameInputGamepadButtons and those into GameInputUiNavigationButtons, checked bit by bit
//! against the values of XInput.h and the GDK's GameInput.h written here as literals.

use xcl::hle::gameinput::{
    buttons_from_xinput, ui_navigation, A, B, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, DPAD_UP, LEFT_SHOULDER, LEFT_STICK,
    MENU, RIGHT_SHOULDER, RIGHT_STICK, VIEW, X, Y,
};

/// XInput.h XINPUT_GAMEPAD_* -> GameInput.h GameInputGamepad* for the same button.
const XINPUT_TO_GAMEPAD: [(u16, u32); 14] = [
    (0x0001, 0x0040), // DPAD_UP -> DPadUp
    (0x0002, 0x0080), // DPAD_DOWN -> DPadDown
    (0x0004, 0x0100), // DPAD_LEFT -> DPadLeft
    (0x0008, 0x0200), // DPAD_RIGHT -> DPadRight
    (0x0010, 0x0001), // START -> Menu
    (0x0020, 0x0002), // BACK -> View
    (0x0040, 0x1000), // LEFT_THUMB -> LeftThumbstick
    (0x0080, 0x2000), // RIGHT_THUMB -> RightThumbstick
    (0x0100, 0x0400), // LEFT_SHOULDER -> LeftShoulder
    (0x0200, 0x0800), // RIGHT_SHOULDER -> RightShoulder
    (0x1000, 0x0004), // A
    (0x2000, 0x0008), // B
    (0x4000, 0x0010), // X
    (0x8000, 0x0020), // Y
];

/// GameInputGamepad* -> GameInputUiNavigation* (Menu 0x1, View 0x2, Accept 0x4, Cancel 0x8, Up 0x10,
/// Down 0x20, Left 0x40, Right 0x80, Context1 0x100, Context2 0x200, PageLeft 0x4000,
/// PageRight 0x8000), the Xbox gamepad's UI navigation layout.
const GAMEPAD_TO_UI: [(u32, u32); 12] = [
    (0x0001, 0x0001),
    (0x0002, 0x0002),
    (0x0004, 0x0004),
    (0x0008, 0x0008),
    (0x0040, 0x0010),
    (0x0080, 0x0020),
    (0x0100, 0x0040),
    (0x0200, 0x0080),
    (0x0010, 0x0100),
    (0x0020, 0x0200),
    (0x0400, 0x4000),
    (0x0800, 0x8000),
];

/// UI navigation Context3 / Context4: what a thumbstick click may give; not pinned here.
const UI_CONTEXT_3_4: u32 = 0x0400 | 0x0800;

/// The exported GameInputGamepadButtons constants have GameInput.h's values.
#[test]
fn gamepad_button_constants_match_gameinput_h() {
    let ours = [
        MENU, VIEW, A, B, X, Y, DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_STICK,
        RIGHT_STICK,
    ];
    let header: Vec<u32> = (0..14).map(|i| 1 << i).collect();
    assert_eq!(ours.to_vec(), header);
}

/// Every one of the 65536 XInput button words maps each pressed button to its GameInput bit and
/// nothing else; 0x0400 and 0x0800 (no XInput.h button; 0x0400 is the guide button of the
/// undocumented XInputGetStateEx) give nothing.
#[test]
fn every_xinput_button_word_maps_bit_by_bit() {
    for w in 0..=u16::MAX {
        let want = XINPUT_TO_GAMEPAD.iter().filter(|(x, _)| w & x != 0).fold(0, |b, (_, g)| b | g);
        assert_eq!(buttons_from_xinput(w), want, "XInput buttons {w:#06x}");
    }
    assert_eq!(buttons_from_xinput(0x0400 | 0x0800), 0);
    assert_eq!(buttons_from_xinput(u16::MAX), 0x3FFF, "all fourteen buttons");
}

/// Every combination of the fourteen gamepad buttons gives the UI navigation bits of the buttons
/// pressed; the thumbstick clicks add nothing else, and bits above the gamepad's are ignored.
#[test]
fn every_gamepad_combination_maps_to_ui_navigation() {
    for b in 0..1u32 << 14 {
        let want = GAMEPAD_TO_UI.iter().filter(|(g, _)| b & g != 0).fold(0, |u, (_, n)| u | n);
        let got = ui_navigation(b);
        assert_eq!(got & !UI_CONTEXT_3_4, want, "gamepad buttons {b:#06x}");
        if b & (LEFT_STICK | RIGHT_STICK) == 0 {
            assert_eq!(got, want, "gamepad buttons {b:#06x}");
        }
        assert_eq!(ui_navigation(b | 0xFFFF_C000), got, "high bits with {b:#06x}");
    }
}

/// A title's menu reads XInput through both maps: A accepts, B cancels, START is Menu, BACK is
/// View, the D-pad moves and the shoulders page.
#[test]
fn xinput_to_ui_navigation_for_menus() {
    let cases: [(u16, u32); 9] = [
        (0x1000, 0x0004),
        (0x2000, 0x0008),
        (0x0010, 0x0001),
        (0x0020, 0x0002),
        (0x0001 | 0x0008, 0x0010 | 0x0080),
        (0x0002 | 0x0004, 0x0020 | 0x0040),
        (0x0100, 0x4000),
        (0x0200, 0x8000),
        (0x4000 | 0x8000, 0x0100 | 0x0200),
    ];
    for (xinput, ui) in cases {
        assert_eq!(ui_navigation(buttons_from_xinput(xinput)), ui, "XInput {xinput:#06x}");
    }
    assert_eq!(ui_navigation(buttons_from_xinput(0)), 0);
}
