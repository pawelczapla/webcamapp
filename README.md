
Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@pawelczapla 
letmaik
/
pyvirtualcam
3
9519
Code
Issues
Pull requests
Discussions
Actions
Security
Insights
You’re making changes in a project you don’t have write access to. We’ve created a fork of this project for you to commit your proposed changes to. Submitting a change will write it to a new branch in your fork, so you can send a pull request.
pyvirtualcam
/
README.md
in
letmaik:master
 

Spaces

4

Soft wrap
1
# pyvirtualcam
2
​
3
pyvirtualcam sends frames to a virtual camera from Python.
4
​
5
## Usage
6
​
7
```py
8
import pyvirtualcam
9
import numpy as np
10
​
11
with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
12
    print(f'Using virtual camera: {cam.device}')
13
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
14
    while True:
15
        frame[:] = cam.frames_sent % 255  # grayscale animation
16
        cam.send(frame)
17
        cam.sleep_until_next_frame()
18
```
19
​
20
pyvirtualcam uses the first available virtual camera it finds (see later section).
21
​
22
For more examples, including using different pixel formats like BGR, or selecting a specific camera device, check out the [`samples/`](samples) folder.
23
​
24
See also the [API Documentation](https://letmaik.github.io/pyvirtualcam).
25
​
26
## Installation
27
​
28
This package works on Windows, macOS, and Linux. Install it from PyPI with:
29
​
30
```sh
31
pip install pyvirtualcam
32
```
33
​
34
pyvirtualcam relies on existing virtual cameras which have to be installed first. See the next section for details.
35
​
36
## Supported virtual cameras
37
​
38
### Windows: OBS
39
​
40
[OBS](https://obsproject.com/) includes a built-in virtual camera for Windows (since 26.0).
41
​
42
To use the OBS virtual camera, simply [install OBS](https://obsproject.com/).
43
​
44
Note that OBS provides a single camera instance only, so it is *not* possible to send frames from Python to the built-in OBS virtual camera, capture the camera in OBS, mix it with other content, and output it again to OBS' built-in virtual camera. To achieve such a workflow, use another virtual camera from Python (like Unity Capture) so that OBS' built-in virtual camera is free for use in OBS.
45
​
46
### Windows: Unity Capture
47
​
48
[Unity Capture](https://github.com/schellingb/UnityCapture) provides a virtual camera originally meant for streaming Unity games. Compared to most other virtual cameras it supports RGBA frames (frames with transparency) which in turn can be captured in [OBS](https://obsproject.com/) for further processing.
49
​
50
To use the Unity Capture virtual camera, follow the [installation instructions](https://github.com/schellingb/UnityCapture#installation) on the project site.
51
​
52
### macOS: OBS
53
​
54
[OBS](https://obsproject.com/) includes a built-in virtual camera for macOS (since 26.1).
55
​
56
To use the OBS virtual camera, follow these one-time setup steps:
57
- [Install OBS](https://obsproject.com/).
58
- Start OBS.
Nie wybrano pliku
Attach files by dragging & dropping, selecting or pasting them.
@pawelczapla
Propose changes
Commit summary
Create README.md
Optional extended description
Add an optional extended description…
 
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
