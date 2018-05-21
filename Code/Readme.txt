Readme

This project's scope is currently restricted to offline recorded videos.
1. Place pre-recorded video's in the ~\Data\ folder.
2. Install opencv and python 3.5
3. Run command "python main.py <Path to video file><filename>"
4. View the tracked fruit , distance to fruit and direction parameters on the window displayed.
5. View various intermediate result snapshots in ~\Data\Results
Incase of experimenting, various thresholds can be changed in:
	i. get_color_mask(image)
	ii.apply_morphology(mask)
	iii.get_width()
	iv.get_distance_to_camera(knownWidth, perWidth)
