import keyboard

while True:
    g = keyboard.on_press_key('g', print)
    r = keyboard.on_press_key('r', print)
    keyboard.wait('esc')
    keyboard.unhook(g)
    keyboard.unhook(r)