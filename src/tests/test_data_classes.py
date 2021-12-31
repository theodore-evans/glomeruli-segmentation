from app.data_classes import Rectangle, Vector2


def test_vector2_subscriptable():
    vec = Vector2(1, 2)
    x, y = vec
    assert vec[0] == 1
    assert vec[1] == 2
    assert vec[0] == vec.x
    assert vec[1] == vec.y
    assert x == 1
    assert y == 2


def test_rectangle_constructor_takes_tuple_but_stores_vector2():
    rect = Rectangle(upper_left=(15, 20), width=10, height=10, level=0)
    assert isinstance(rect.upper_left, Vector2)
    assert rect.upper_left.x == 15
    assert rect.upper_left.y == 20
