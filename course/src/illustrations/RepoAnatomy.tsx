// A model repository drawn as its files, each showing a real excerpt (from openai-community/gpt2),
// connected to the part of the familiar pipeline that the file supplies. The accent marks the
// connections: that mapping is what the picture is about.
import { ACC, Caption, FILL, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

const shade = (i: number, j: number, seed: number) => {
  const s = Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 37.719) * 43758.5453
  return s - Math.floor(s)
}

function Sheet({ y, h, name, size, lines }: { y: number; h: number; name: string; size: string; lines: string[] }) {
  return (
    <g>
      <path d={`M14 ${y} H318 L332 ${y + 14} V${y + h} H14 Z`} fill={FILL} stroke={LINE} />
      <path d={`M318 ${y} V${y + 14} H332`} fill="none" stroke={LINE} />
      <text x={24} y={y + 17} fontSize={11.5} fontWeight={700} style={{ fill: INK, ...MONO }}>{name}</text>
      <text x={312} y={y + 17} textAnchor="end" fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{size}</text>
      {lines.map((l, i) => <text key={i} x={24} y={y + 34 + i * 14} fontSize={10.5} style={{ fill: INK, ...MONO, whiteSpace: 'pre', fontVariantLigatures: 'none' }}>{l}</text>)}
    </g>
  )
}

const Link = ({ d }: { d: string }) => <path d={d} fill="none" stroke={ACC} strokeWidth={1.8} markerEnd="url(#repo-arrow-acc)" />

const PARTS = [
  'Tokenizer files (vocab.json and merges.txt; tokenizer.json packs the same information for the fast tokenizer): the text to ids step. 50,000 merges plus 256 bytes plus one special token give 50,257 ids.',
  'config.json: the skeleton of the network. 12 blocks, width 768, 12 heads, 1,024 positions, 50,257 tokens. No weights.',
  'model.safetensors: a JSON header listing every tensor with its type, shape and byte range, then the raw numbers that fill the skeleton.',
  'generation_config.json: default settings for the sampling loop, such as which token id ends generation.',
]

export function RepoAnatomy() {
  const chips: [string, number, number][] = [['The', 30, 464], ['␣cat', 36, 3797], ['␣sat', 36, 3332]]
  let cx = 470
  return (
    <Figure
      id="repo"
      height={452}
      minWidth={700}
      title="A model repository drawn as four groups of files, each connected to the part of the text-to-next-token pipeline it supplies: tokenizer files to the text-to-ids step, config.json to the empty skeleton of the network, the safetensors file to the numbers inside it, and generation_config.json to the sampling step"
      after={<ol className="sr-only">{PARTS.map((p) => <li key={p}>{p}</li>)}</ol>}
    >
      <text x={14} y={14} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>huggingface.co/openai-community/gpt2</text>
      <text x={746} y={14} textAnchor="end" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>the machine you already know</text>

      {/* ---------------- files ---------------- */}
      <Sheet y={26} h={82} name="vocab.json · merges.txt" size="1.5 MB" lines={[
        'merges   Ġ t │ Ġ a │ h e │ i n │ …  50,000',
        'vocab    "The": 464, "Ġcat": 3797, "Ġsat": 3332',
        'special  "<|endoftext|>": 50256',
      ]} />
      <Sheet y={120} h={82} name="config.json" size="665 B" lines={[
        '"model_type": "gpt2",  "n_layer": 12,',
        '"n_head": 12,  "n_embd": 768,',
        '"n_positions": 1024,  "vocab_size": 50257',
      ]} />
      <Sheet y={214} h={138} name="model.safetensors" size="548 MB" lines={[
        'header: name → dtype, shape, byte range',
        ' wte.weight              F32 [50257, 768]',
        ' wpe.weight              F32 [1024, 768]',
        ' h.0.attn.c_attn.weight  F32 [768, 2304]',
        ' h.0.mlp.c_fc.weight     F32 [768, 3072]',
        ' … 160 tensors in all',
        'then: the raw bytes. No code.',
      ]} />
      <Sheet y={364} h={54} name="generation_config.json" size="124 B" lines={['"bos_token_id": 50256,  "eos_token_id": 50256']} />
      <text x={14} y={440} fontSize={11.5} style={{ fill: SOFT }}>Ġ is how GPT-2’s files write a leading space.</text>

      {/* ---------------- 1 tokenizer -> text to ids ---------------- */}
      <Link d="M338 67 H452" />
      <Caption x={395} y={56} n={1} title="text → ids" />
      <text x={470} y={46} fontSize={12} style={{ fill: INK }}>“The cat sat”</text>
      {chips.map(([t, w, id]) => {
        const x = cx
        cx += w + 5
        return (
          <g key={t}>
            <rect x={x} y={56} width={w} height={22} rx={5} fill={FILL} stroke={LINE} />
            <text x={x + w / 2} y={71} textAnchor="middle" fontSize={11} style={{ fill: INK, ...MONO }}>{t}</text>
            <text x={x + w / 2} y={94} textAnchor="middle" fontSize={10.5} style={{ fill: INK, ...MONO }}>{id}</text>
          </g>
        )
      })}
      <text x={600} y={71} fontSize={11.5} style={{ fill: SOFT }}>tokens</text>
      <text x={600} y={94} fontSize={11.5} style={{ fill: SOFT }}>ids</text>
      <path d="M523 100 V122" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#repo-arrow)" />

      {/* ---------------- 2 + 3: the network: skeleton from config, numbers from safetensors ---------------- */}
      <rect x={456} y={128} width={290} height={222} rx={10} fill="none" stroke={INK} strokeWidth={1.4} strokeDasharray="6 4" />
      {/* wte + wpe */}
      {Array.from({ length: 4 }, (_, r) => Array.from({ length: 14 }, (_, c) => (
        <rect key={`${r}-${c}`} x={470 + c * 8} y={140 + r * 7} width={6.6} height={5.6} rx={1.5} fill={INK} opacity={0.16 + shade(r, c, 4) * 0.55} />
      )))}
      <text x={590} y={152} fontSize={10.5} style={{ fill: INK, ...MONO }}>wte  50257 × 768</text>
      <text x={590} y={166} fontSize={10.5} style={{ fill: INK, ...MONO }}>wpe   1024 × 768</text>
      {/* 12 blocks */}
      {Array.from({ length: 12 }, (_, i) => (
        <rect key={i} x={470} y={180 + i * 10.5} width={110} height={7.5} rx={3} fill={INK} opacity={0.18 + shade(i, 1, 9) * 0.4} />
      ))}
      <text x={590} y={192} fontSize={10.5} style={{ fill: INK, ...MONO }}>h.0 … h.11</text>
      <text x={590} y={208} fontSize={11.5} style={{ fill: SOFT }}>12 blocks, each:</text>
      <text x={590} y={223} fontSize={10.5} style={{ fill: INK, ...MONO }}>ln_1 attn.c_attn</text>
      <text x={590} y={237} fontSize={10.5} style={{ fill: INK, ...MONO }}>attn.c_proj ln_2</text>
      <text x={590} y={251} fontSize={10.5} style={{ fill: INK, ...MONO }}>mlp.c_fc mlp.c_proj</text>
      <rect x={470} y={312} width={110} height={9} rx={4.5} fill="none" stroke={INK} strokeWidth={1.1} />
      <text x={590} y={321} fontSize={10.5} style={{ fill: INK, ...MONO }}>ln_f</text>
      <text x={470} y={340} fontSize={10.5} style={{ fill: INK, ...MONO }}>lm_head = wte (tied, not stored twice)</text>

      <Link d="M338 161 H450" />
      <Caption x={395} y={136} n={2} title="the skeleton" />
      <text x={395} y={150} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT }}>how many, how wide</text>
      <Link d="M338 283 H464" />
      <Caption x={397} y={258} n={3} title="the numbers" />
      <text x={397} y={272} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT }}>124,439,808 of them</text>

      {/* ---------------- 4 generation config -> sampling loop ---------------- */}
      <path d="M523 352 V372" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#repo-arrow)" />
      <Link d="M338 391 H452" />
      <Caption x={395} y={380} n={4} title="loop defaults" />
      {[[' floor', 7.6], [' bed', 6.5], [' couch', 5.4]].map(([t, p], i) => (
        <g key={t as string}>
          <text x={508} y={388 + i * 14} textAnchor="end" fontSize={10.5} style={{ fill: INK, ...MONO }}>{t}</text>
          <rect x={514} y={380 + i * 14} width={(p as number) * 7} height={9} rx={3} fill={INK} opacity={0.35} />
          <text x={519 + (p as number) * 7} y={388 + i * 14} fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{p}%</text>
        </g>
      ))}
      <text x={606} y={388} fontSize={11.5} style={{ fill: SOFT }}>sample, append, repeat</text>
      <text x={606} y={403} fontSize={11.5} style={{ fill: SOFT }}>until id 50256 or a limit</text>
    </Figure>
  )
}
