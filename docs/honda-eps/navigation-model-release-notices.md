# Corrected Bigv2 release notices

These notices accompany `bigv2-amd-map-fp32.sptg` and the supporting compiler,
warp and provenance files distributed for `big-nav-v2-map-fp32`.

The published device-created `bigv2-sidecars-from-comma.zip` contains seven
registration/provenance files and no license texts. Its bytes remain unchanged.
Distributors must supply this notice alongside those assets; the installer keeps
this document with the installed package. The release should also retain these
notices (as a companion notice asset or in its release description).

## License scope and source evidence

- The source comma model is identified by ONNX SHA256
  `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
  [The exact official checkpoint pointer](https://github.com/commaai/openpilot/blob/a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx)
  names that SHA256, and [comma's license at that revision](https://github.com/commaai/openpilot/blob/a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9/LICENSE)
  is MIT. [The official model-hosting card](https://huggingface.co/commaai/openpilot-lfs/blob/51b12d20a25b76fa8a5717c2e5af01f148e469e2/README.md)
  also declares `license: mit`.
- The pinned tinygrad runtime, including code used to generate native model/warp
  programs, is revision `69fd4d34d4430036378aba2b43f9d339cdd6ea73`, under
  [tinygrad's MIT license](https://github.com/tinygrad/tinygrad/blob/69fd4d34d4430036378aba2b43f9d339cdd6ea73/LICENSE).
- The custom SunnyPilot integration and its retained compiler/provenance source
  carry the branch's distinct custom license below. They are not all covered by
  comma's or tinygrad's MIT license. The custom license requires written
  permission for commercial, for-profit or closed-source uses.

This software is licensed under a custom license requiring permission for use.

This project uses software from Haibin Wen and SUNNYPILOT LLC and is licensed under a custom license requiring permission for use.

## Comma.ai MIT notice

Copyright (c) 2018, Comma.ai, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## tinygrad MIT notice

Copyright (c) 2024, the tiny corp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## SunnyPilot custom license

# Custom MIT License

Copyright (c) 2024, Haibin Wen, SUNNYPILOT LLC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to view and modify the Software, subject to the following conditions:

1. **Permission Required**: Permission Required for Commercial, For-Profit, or Closed Source Use: Use of the Software, in whole or in part, for any commercial purposes, for-profit projects, or in closed source projects requires explicit written permission from the original author(s).

2. **Redistribution**: Any redistribution of the Software, modified or unmodified, must retain this license notice and the following acknowledgment:
   "This software is licensed under a custom license requiring permission for use."

3. **Visibility**: Any project that uses the Software must visibly mention the following acknowledgment:
   "This project uses software from Haibin Wen and SUNNYPILOT LLC and is licensed under a custom license requiring permission for use."

4. **No Warranty**: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contact sunnypilot Support <support@sunnypilot.ai> for permission requests.

---

Haibin Wen, SUNNYPILOT LLC
