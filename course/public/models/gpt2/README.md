# GPT-2 small, quantized for this course

These files are **GPT-2 small (124M parameters) by OpenAI**, released under the modified MIT license
from https://github.com/openai/gpt-2 (reproduced below). They were downloaded from the Hugging Face copy
`openai-community/gpt2` and **quantized for this course** by `phase6-engineering/export_gpt2.py`:
every weight matrix is stored as int8 with one float32 scale per output row; LayerNorm parameters and
biases stay float32. The token embedding (which is also the output head) keeps 8 "outlier" columns in
float32, because GPT-2's final hidden state is huge in those few dimensions. That changes the weights slightly (see the "Making models cheaper" lesson), so the
model here is a close approximation of GPT-2, not bit-identical to it.

- `gpt2-int8-*.bin`: 4 weight chunks (each at most 40 MB), described by `manifest.json`
- `merges.bin`: the GPT-2 byte-level BPE merges as uint16 id pairs (the vocabulary follows from them)
- total about 127 MB

Content generated with these weights is GPT-2 output. GPT-2 was trained on web text (WebText, 2019) and
can produce text that is wrong, biased or offensive.

## License of the original weights

```
Modified MIT License

Software Copyright (c) 2019 OpenAI

We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```
