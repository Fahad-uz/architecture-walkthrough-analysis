from architecture_walkthrough.geometry.models import Point2D, RoomPolygon
from architecture_walkthrough.geometry.room_extraction import missing_room_label_issues
from architecture_walkthrough.vision.ocr import OCRText


def label(name, x, y, semantic_type="room_label"):
    return OCRText(name, [(x-1,y-1),(x+1,y-1),(x+1,y+1),(x-1,y+1)], .98, name, semantic_type)


def test_missing_living_room_is_reported_without_closing_its_entrance():
    bedroom = RoomPolygon(name="Bedroom", points=[Point2D(x=0,y=0),Point2D(x=3,y=0),Point2D(x=3,y=3),Point2D(x=0,y=3)])
    labels = [label("Bedroom", 100, 900), label("Living and Dining", 600, 700)]
    issues = missing_room_label_issues([bedroom], labels, 100, 1000)
    assert len(issues) == 1
    assert issues[0].code == "unreconstructed_labeled_room"
    assert "Living and Dining" in issues[0].message
    assert len(bedroom.points) == 4


def test_dimension_and_duplicate_labels_do_not_create_extra_missing_rooms():
    labels = [label("Study room", 200, 300), label("Study room", 200, 300),
              label("10' x 12'", 300, 500), label("stairs", 700, 500, "stair_label")]
    issues = missing_room_label_issues([], labels, 100, 1000)
    assert len(issues) == 1
    assert "Study room" in issues[0].message
