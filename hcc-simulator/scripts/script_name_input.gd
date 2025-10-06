extends LineEdit

signal output(filename)

func _on_run_button_pressed() -> void:
	output.emit(self.text)
