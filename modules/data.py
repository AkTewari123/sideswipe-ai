import math


def _joystick_action(detections):
    print("Calculating joystick action...")
    names = detections.data["class_name"]
    joy_center = None
    print("names: ", names)
    for i, n in enumerate(names):
        if n == "joystick":
            print(f"Found joystick at index {i}")
            x1, y1, x2, y2 = detections.xyxy[i]
            joy_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if joy_center is None:
        return [0.0, 0.0, 0.0]
    ang = find_joystick_angle(joy_center)
    print(f"joystick angle: {ang}")
    if ang is None:
        return [0.0, 0.0, 0.0]
    return [1.0, math.sin(ang), math.cos(ang)]


def find_joystick_angle(joystick_center, dead_zone=75):
    JOYSTICK_ABS_CENTER = (530, 1130)
    """
    Angle of the joystick nub relative to the joystick's center circle,
    expressed on a math-convention unit circle centered at circle_center:
        0      = right
        pi/2   = up
        pi     = left
        3*pi/2 = down
    Returned in radians in the range [0, 2*pi).

    +y is UP (image y is flipped), matching the rest of the ref2 pipeline.
    Returns None if the nub is within the dead-zone (~centered / not pushed).
    """
    dx = joystick_center[0] - JOYSTICK_ABS_CENTER[0]
    dy = -(joystick_center[1] - JOYSTICK_ABS_CENTER[1])  # flip y so +y = up
    dist = math.hypot(dx, dy)
    if dist <= dead_zone:  # dead-zone: no meaningful direction
        print(f"dist: {dist}, dead_zone: {dead_zone}")
        return None

    ang = math.atan2(dy, dx)
    if ang < 0:
        ang += 2 * math.pi  # map -pi..0 into pi..2pi -> [0, 2pi)
    return ang
