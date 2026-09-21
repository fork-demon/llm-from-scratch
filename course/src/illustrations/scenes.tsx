// One cartoon scene per part of the course (plus the home hero). Keyed by part id.
import type { ReactNode } from 'react'
import { Bubble, C, Chip, Cloud, Scene, Sparkle, Tok, freeText, outlined, sceneText } from './kit'

/* Part 0: a question goes in, tokens come out one at a time */
function BigPicture() {
  return (
    <Scene label="Tok the robot reads the question “What is a cat?” and answers one token at a time">
      <Bubble x={24} y={28} w={170} h={50} tail="right">
        <text x={109} y={59} textAnchor="middle" {...sceneText(16)}>What is a cat?</text>
      </Bubble>
      <Tok x={150} y={272} look={[3, -1]} armR={[78, -96]} armL={[-60, -40]} />
      {/* tokens streaming out */}
      <path d="M236 170 C 280 120, 320 120, 440 96" fill="none" stroke={C.ink} strokeWidth={2.5} strokeDasharray="2 9" strokeLinecap="round" />
      <Chip x={262} y={150} w={44} text="A" fill={C.amber} rotate={-8} />
      <Chip x={318} y={124} w={56} text="cat" fill={C.mint} rotate={-6} />
      <Chip x={378} y={108} w={48} text="is" fill={C.sky} rotate={-4} />
      <Chip x={432} y={96} w={44} text="a" fill={C.coral} rotate={-2} />
      <g opacity="0.45"><Chip x={400} y={178} w={72} text="small" rotate={4} /></g>
      <g opacity="0.25"><Chip x={330} y={214} w={64} text="furry" rotate={-3} /></g>
      <text x={368} y={262} textAnchor="middle" {...freeText(12.5, 500)}>one token at a time</text>
      <Sparkle x={222} y={118} s={0.9} />
      <Sparkle x={452} y={150} s={0.6} fill={C.white} />
    </Scene>
  )
}

/* Part 0 card: lifting the lid off the black box */
function OpenBox() {
  return (
    <Scene label="Tok lifts the lid off a black box and finds an orderly row of steps inside">
      {/* lid, tilted open */}
      <g transform="rotate(-16 250 96)"><rect x={222} y={70} width={216} height={26} rx={8} fill={C.ink} {...outlined} /><rect x={310} y={58} width={40} height={14} rx={6} fill={C.ink} {...outlined} /></g>
      {/* box */}
      <rect x={232} y={128} width={204} height={128} rx={12} fill={C.ink} {...outlined} />
      <rect x={244} y={118} width={180} height={22} rx={8} fill={C.white} {...outlined} />
      {/* what is inside */}
      <Chip x={290} y={170} w={76} h={26} text="tokens" fill={C.amber} mono={false} size={12} />
      <Chip x={380} y={170} w={76} h={26} text="vectors" fill={C.mint} mono={false} size={12} />
      <Chip x={290} y={206} w={76} h={26} text="attention" fill={C.sky} mono={false} size={12} />
      <Chip x={380} y={206} w={76} h={26} text="scores" fill={C.coral} mono={false} size={12} />
      <Chip x={335} y={240} w={120} h={24} text="pick one, repeat" fill={C.white} mono={false} size={11.5} />
      <text x={334} y={284} textAnchor="middle" {...freeText(12.5, 500)}>no magic inside, only steps</text>
      <Tok x={120} y={274} s={0.95} look={[4, -2]} mood="wow" armR={[92, -150]} armL={[-60, -44]} />
      <Sparkle x={214} y={52} s={0.9} />
      <Sparkle x={452} y={104} s={0.6} fill={C.white} />
    </Scene>
  )
}

/* Part 1: two arrows and the angle between them */
function MathScene() {
  return (
    <Scene label="Tok holds two arrows on a grid and compares their directions">
      <g stroke={C.soft} strokeWidth={2}>
        {[210, 250, 290, 330, 370, 410, 450].map((x) => <line key={x} x1={x} y1={40} x2={x} y2={250} />)}
        {[50, 90, 130, 170, 210, 250].map((y) => <line key={y} x1={200} y1={y} x2={460} y2={y} />)}
      </g>
      <line x1={210} y1={250} x2={460} y2={250} {...outlined} />
      <line x1={210} y1={250} x2={210} y2={40} {...outlined} />
      {/* vectors */}
      <g {...outlined} strokeWidth={6} fill="none">
        <path d="M210 250 L410 110" stroke={C.ink} strokeWidth={10} />
        <path d="M210 250 L410 110" stroke={C.coral} />
        <path d="M210 250 L440 196" stroke={C.ink} strokeWidth={10} />
        <path d="M210 250 L440 196" stroke={C.sky} />
      </g>
      <path d="M410 110 l-22 2 l12 17z" fill={C.coral} {...outlined} />
      <path d="M440 196 l-20 -6 l4 20z" fill={C.sky} {...outlined} />
      <path d="M290 231 A 82 82 0 0 0 277 203" fill="none" {...outlined} />
      <Chip x={335} y={222} w={70} text="agree?" fill={C.amber} mono={false} size={13} />
      <text x={420} y={96} {...freeText(15, 700)}>a</text>
      <text x={450} y={186} {...freeText(15, 700)}>b</text>
      <Tok x={104} y={274} s={0.92} look={[4, -2]} mood="think" armR={[70, -110]} armL={[-58, -44]} />
      <Sparkle x={178} y={58} s={0.8} />
    </Scene>
  )
}

/* Part 2: walking downhill in fog */
function Learning() {
  return (
    <Scene label="Tok walks downhill through fog toward a flag at the bottom of a valley" blobs={false}>
      <path d="M-10 96 C 90 100, 150 250, 250 250 C 340 250, 380 150, 490 120 L490 310 L-10 310Z" fill={C.mint} {...outlined} />
      {/* footsteps */}
      <g fill={C.ink} opacity="0.35">
        {[[70, 128], [96, 158], [124, 190], [156, 218], [190, 238]].map(([x, y], i) => <ellipse key={i} cx={x} cy={y} rx={6} ry={3.5} transform={`rotate(38 ${x} ${y})`} />)}
      </g>
      {/* flag at the minimum */}
      <line x1={252} y1={250} x2={252} y2={188} {...outlined} />
      <path d="M252 190 L292 202 L252 216Z" fill={C.coral} {...outlined} />
      <text x={252} y={276} textAnchor="middle" {...sceneText(12.5, 700)}>lowest loss</text>
      <Tok x={128} y={196} s={0.62} look={[4, 3]} armR={[60, -30]} armL={[-56, -70]} />
      <path d="M176 176 l26 22" {...outlined} strokeDasharray="1 8" />
      <path d="M206 202 l-13 -3 l6 -10z" fill={C.ink} />
      <Cloud x={330} y={86} s={1.25} />
      <Cloud x={92} y={56} s={0.9} opacity={0.8} />
      <Cloud x={420} y={190} s={0.85} opacity={0.7} />
      <Chip x={392} y={40} w={128} text="step = −lr × slope" fill={C.amber} size={11.5} />
    </Scene>
  )
}

/* Part 3: plugging a neural network together */
function NeuralNets() {
  const layers: [number, number[]][] = [[230, [90, 150, 210]], [320, [60, 120, 180, 240]], [410, [110, 190]]]
  const lit = new Set(['0-1', '1-1', '1-2', '2-0'])
  return (
    <Scene label="Tok connects a small network of neurons; some of them light up">
      <g stroke={C.ink} strokeWidth={2} opacity="0.4">
        {layers.slice(0, -1).flatMap(([x1, ys1], li) => ys1.flatMap((y1) => layers[li + 1][1].map((y2) => <line key={`${li}-${y1}-${y2}`} x1={x1} y1={y1} x2={layers[li + 1][0]} y2={y2} />)))}
      </g>
      <g stroke={C.amber} strokeWidth={5} strokeLinecap="round">
        <line x1={230} y1={150} x2={320} y2={120} /><line x1={230} y1={150} x2={320} y2={180} /><line x1={320} y1={120} x2={410} y2={110} /><line x1={320} y1={180} x2={410} y2={110} />
      </g>
      {layers.map(([x, ys], li) => ys.map((y, ni) => (
        <circle key={`${li}-${ni}`} cx={x} cy={y} r={15} fill={lit.has(`${li}-${ni}`) ? C.amber : C.white} {...outlined} />
      )))}
      <Sparkle x={436} y={84} s={0.8} />
      {/* cable from Tok's hand to the input */}
      <path d="M150 162 C 176 120, 196 170, 215 152" fill="none" {...outlined} strokeWidth={4} />
      <rect x={140} y={154} width={16} height={12} rx={3} fill={C.coral} {...outlined} />
      <Tok x={92} y={274} s={0.92} look={[4, -1]} armR={[62, -118]} armL={[-58, -44]} mood="wink" />
      <Chip x={410} y={264} w={86} text="prediction" fill={C.mint} mono={false} size={12} />
      <path d="M410 214 L410 244" {...outlined} /><path d="M410 250 l-6 -10 h12z" fill={C.ink} />
    </Scene>
  )
}

/* Part 4: cutting a word into tokens */
function TextToNumbers() {
  return (
    <Scene label="Tok cuts the word “unbelievable” into three tokens and each one gets a number">
      <rect x={176} y={52} width={280} height={46} rx={14} fill={C.white} {...outlined} />
      <text x={316} y={83} textAnchor="middle" {...sceneText(22, 700)} style={{ fill: 'var(--il-chip-text)', fontFamily: 'var(--mono)' }}>unbelievable</text>
      <g {...outlined} strokeDasharray="5 6" stroke={C.coral}><line x1={243} y1={44} x2={243} y2={106} /><line x1={367} y1={44} x2={367} y2={106} /></g>
      {/* scissors */}
      <g transform="translate(243 120) rotate(-8)">
        <path d="M0 -6 L-8 30 M0 -6 L8 30" {...outlined} strokeWidth={4} />
        <circle cx={-10} cy={38} r={8} fill={C.coral} {...outlined} /><circle cx={10} cy={38} r={8} fill={C.coral} {...outlined} />
      </g>
      <Chip x={224} y={196} w={64} h={36} text="un" fill={C.amber} size={16} rotate={-5} />
      <Chip x={316} y={204} w={96} h={36} text="believ" fill={C.mint} size={16} />
      <Chip x={412} y={196} w={78} h={36} text="able" fill={C.sky} size={16} rotate={5} />
      <g {...outlined}><path d="M224 220 v18" /><path d="M316 228 v14" /><path d="M412 220 v18" /></g>
      <text x={224} y={262} textAnchor="middle" {...freeText(15, 700)} style={{ fill: C.text, fontFamily: 'var(--mono)' }}>403</text>
      <text x={316} y={266} textAnchor="middle" {...freeText(15, 700)} style={{ fill: C.text, fontFamily: 'var(--mono)' }}>11726</text>
      <text x={412} y={262} textAnchor="middle" {...freeText(15, 700)} style={{ fill: C.text, fontFamily: 'var(--mono)' }}>540</text>
      <Tok x={92} y={276} s={0.92} look={[4, -2]} armR={[70, -92]} armL={[-58, -44]} />
      <Sparkle x={160} y={40} s={0.7} />
    </Scene>
  )
}

/* Part 5: guessing the next word */
function FirstLm() {
  const opts: [string, number, string][] = [['mat', 118, C.mint], ['sofa', 64, C.sky], ['floor', 40, C.amber], ['moon', 10, C.coral]]
  return (
    <Scene label="Tok rolls a die to choose the next word after “The cat sat on the”, with “mat” the most likely">
      <rect x={150} y={34} width={304} height={46} rx={14} fill={C.white} {...outlined} />
      <text x={166} y={64} {...sceneText(17, 600)}>The cat sat on the</text>
      <rect x={366} y={44} width={74} height={26} rx={8} fill={C.panel} {...outlined} strokeDasharray="5 5" />
      <text x={403} y={63} textAnchor="middle" {...freeText(15, 700)}>?</text>
      {opts.map(([w, len, fill], i) => (
        <g key={w}>
          <text x={262} y={128 + i * 34} textAnchor="end" {...freeText(14, 600)} style={{ fill: C.text, fontFamily: 'var(--mono)' }}>{w}</text>
          <rect x={272} y={114 + i * 34} width={len} height={18} rx={9} fill={fill} {...outlined} />
        </g>
      ))}
      {/* die */}
      <g transform="translate(196 214) rotate(-14)">
        <rect x={-22} y={-22} width={44} height={44} rx={10} fill={C.white} {...outlined} />
        {[[-10, -10], [10, 10], [0, 0], [-10, 10], [10, -10]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r={3.6} fill={C.ink} />)}
      </g>
      <path d="M168 176 q10 -16 24 -6" fill="none" {...outlined} strokeDasharray="1 7" />
      <Tok x={84} y={276} s={0.9} look={[4, 1]} mood="wow" armR={[74, -70]} armL={[-58, -44]} />
      <Sparkle x={424} y={118} s={0.8} />
    </Scene>
  )
}

/* Part 6: attention = looking at the right words */
function AttentionScene() {
  const words: [string, number, number][] = [['The', 36, 46], ['animal', 110, 78], ['was', 186, 50], ['tired', 250, 64], ['so', 312, 40], ['it', 360, 40]]
  const cx = (i: number) => words[i][1] + words[i][2] / 2
  return (
    <Scene label="The word “it” looks back at the other words; its strongest link goes to “animal”">
      {/* links from "it" */}
      <g fill="none" strokeLinecap="round">
        <path d={`M${cx(5)} 66 C ${cx(5)} 8, ${cx(1)} 0, ${cx(1)} 66`} stroke={C.ink} strokeWidth={12} />
        <path d={`M${cx(5)} 66 C ${cx(5)} 8, ${cx(1)} 0, ${cx(1)} 66`} stroke={C.amber} strokeWidth={7} />
        <path d={`M${cx(5)} 66 C ${cx(5)} 34, ${cx(3)} 30, ${cx(3)} 66`} stroke={C.amber} strokeWidth={3.5} opacity="0.9" />
        <path d={`M${cx(5)} 66 C ${cx(5)} 24, ${cx(0)} 14, ${cx(0)} 66`} stroke={C.amber} strokeWidth={1.5} opacity="0.7" />
      </g>
      {words.map(([w, x, width], i) => <Chip key={w} x={x + width / 2} y={84} w={width} h={34} text={w} mono={false} size={15} fill={i === 5 ? C.sky : i === 1 ? C.amber : C.white} />)}
      <Chip x={cx(1)} y={126} w={54} h={24} text="72%" fill={C.white} size={12} />
      <Chip x={cx(3)} y={126} w={54} h={24} text="19%" fill={C.white} size={12} />
      {/* Tok with a magnifier */}
      <Tok x={330} y={282} s={0.88} flip look={[3, -3]} mood="think" armR={[74, -128]} armL={[-58, -44]} />
      <g transform="translate(236 150)">
        <circle cx={0} cy={0} r={24} fill={C.white} fillOpacity={0.55} {...outlined} strokeWidth={5} />
        <path d="M17 17 L30 30" {...outlined} strokeWidth={7} />
        <text x={0} y={5} textAnchor="middle" {...sceneText(13, 700)}>Q·K</text>
      </g>
      <g transform="translate(60 196)">
        {[['query', C.sky], ['key', C.amber], ['value', '#c4a5ff']].map(([t, f], i) => <Chip key={t} x={40} y={i * 32} w={84} h={26} text={t} fill={f} mono={false} size={12.5} />)}
      </g>
      <Sparkle x={448} y={40} s={0.8} />
    </Scene>
  )
}

/* Part 7: stacking the GPT tower */
function Gpt() {
  const blocks: [string, string][] = [['tokens + positions', C.sky], ['block 1', C.white], ['block 2', C.white], ['block N', C.white], ['next-token scores', C.mint]]
  return (
    <Scene label="Tok stacks blocks into a tower: embeddings at the bottom, Transformer blocks, and next-token scores on top">
      {blocks.map(([t, fill], i) => {
        const y = 250 - i * 42
        const top = i === blocks.length - 1
        return (
          <g key={t} transform={top ? 'translate(18 -30) rotate(-7 330 60)' : undefined}>
            <rect x={230} y={y - 34} width={200} height={34} rx={10} fill={fill} {...outlined} />
            {fill === C.white && <><rect x={240} y={y - 26} width={54} height={18} rx={6} fill={C.amber} {...outlined} strokeWidth={2} /><rect x={300} y={y - 26} width={54} height={18} rx={6} fill={C.coral} {...outlined} strokeWidth={2} /></>}
            <text x={fill === C.white ? 392 : 330} y={y - 12} textAnchor="middle" {...sceneText(12.5, 700)}>{t}</text>
          </g>
        )
      })}
      <text x={330} y={284} textAnchor="middle" {...freeText(12, 500)}>attention + MLP, repeated</text>
      <Tok x={108} y={276} s={0.92} look={[4, -3]} armR={[84, -150]} armL={[-58, -44]} mood="happy" />
      <Sparkle x={204} y={70} s={0.8} />
      <Sparkle x={452} y={150} s={0.6} fill={C.white} />
    </Scene>
  )
}

/* Part 8: upgrades */
function Modern() {
  return (
    <Scene label="Tok flies with a jetpack past badges for modern upgrades: RoPE, GQA, KV cache and SwiGLU">
      <Cloud x={92} y={230} s={1.1} opacity={0.8} />
      <Cloud x={400} y={248} s={0.9} opacity={0.7} />
      <g transform="rotate(-12 200 170)">
        {/* jetpack */}
        <rect x={140} y={120} width={30} height={66} rx={12} fill={C.coral} {...outlined} />
        <path d="M145 186 q10 34 20 0z" fill={C.amber} {...outlined} />
        <Tok x={200} y={232} s={0.88} look={[4, -3]} mood="wow" armR={[80, -120]} armL={[-64, -90]} />
      </g>
      <Chip x={372} y={62} w={92} h={34} text="RoPE" fill={C.sky} size={15} rotate={6} />
      <Chip x={412} y={116} w={84} h={34} text="GQA" fill={C.amber} size={15} rotate={-5} />
      <Chip x={366} y={170} w={116} h={34} text="KV cache" fill={C.mint} size={15} rotate={4} />
      <Chip x={96} y={60} w={104} h={34} text="SwiGLU" fill={C.white} size={15} rotate={-7} />
      <Sparkle x={300} y={40} s={0.9} />
      <Sparkle x={452} y={206} s={0.6} fill={C.white} />
      <Sparkle x={40} y={130} s={0.6} />
    </Scene>
  )
}

/* Part 9: documents, tools, a loop */
function Systems() {
  return (
    <Scene label="Tok at the centre of a loop: fetching documents from a shelf and using tools">
      {/* shelf */}
      <rect x={28} y={50} width={126} height={150} rx={12} fill={C.white} {...outlined} />
      <line x1={28} y1={124} x2={154} y2={124} {...outlined} />
      {[[40, C.coral, 56], [60, C.amber, 50], [80, C.sky, 60], [104, C.mint, 46], [124, C.coral, 54]].map(([x, f, h], i) => <rect key={i} x={x as number} y={120 - (h as number)} width={16} height={h as number} rx={3} fill={f as string} {...outlined} strokeWidth={2.5} />)}
      {[[44, C.sky, 50], [66, C.mint, 58], [96, C.amber, 44]].map(([x, f, h], i) => <rect key={i} x={x as number} y={196 - (h as number)} width={18} height={h as number} rx={3} fill={f as string} {...outlined} strokeWidth={2.5} />)}
      <text x={91} y={222} textAnchor="middle" {...freeText(12.5, 700)}>your documents</text>
      {/* loop */}
      <path d="M190 84 A 118 96 0 1 1 186 222" fill="none" {...outlined} strokeDasharray="2 9" />
      <path d="M186 222 l14 -2 l-6 -13z" fill={C.ink} />
      <Tok x={300} y={266} s={0.84} look={[-4, -1]} armL={[-82, -112]} armR={[78, -100]} />
      {/* a page in the left hand */}
      <g transform="translate(222 150) rotate(-10)"><rect x={-16} y={-22} width={32} height={42} rx={5} fill={C.white} {...outlined} /><g stroke={C.ink} strokeWidth={2} strokeLinecap="round"><line x1={-9} y1={-10} x2={9} y2={-10} /><line x1={-9} y1={-2} x2={9} y2={-2} /><line x1={-9} y1={6} x2={4} y2={6} /></g></g>
      {/* wrench in the right hand */}
      <g transform="translate(372 160) rotate(30)"><rect x={-4} y={-4} width={8} height={40} rx={4} fill={C.soft} {...outlined} /><path d="M-12 -14 a12 12 0 1 0 24 0 l-7 0 l0 8 l-10 0 l0 -8z" fill={C.soft} {...outlined} /></g>
      <Chip x={420} y={70} w={86} text="search" fill={C.sky} mono={false} size={12.5} />
      <Chip x={432} y={230} w={74} text="tools" fill={C.amber} mono={false} size={12.5} />
      <Chip x={300} y={36} w={150} h={26} text="observe → decide → act" fill={C.white} mono={false} size={11.5} />
    </Scene>
  )
}

/* Part 10: running it for real */
function Engineering() {
  const bars = [38, 52, 30, 64, 46, 70, 58]
  return (
    <Scene label="Tok watches a serving dashboard: requests queue up on the left, a latency chart and an eval scorecard fill the screen">
      {/* monitor */}
      <rect x={196} y={40} width={262} height={172} rx={14} fill={C.white} {...outlined} />
      <rect x={300} y={212} width={54} height={20} fill={C.soft} {...outlined} />
      <rect x={268} y={230} width={118} height={12} rx={6} fill={C.soft} {...outlined} />
      {/* latency bars */}
      {bars.map((h, i) => <rect key={i} x={214 + i * 18} y={130 - h} width={12} height={h} rx={3} fill={i === 5 ? C.coral : C.sky} {...outlined} strokeWidth={2} />)}
      <line x1={210} y1={132} x2={344} y2={132} {...outlined} />
      <text x={212} y={150} {...sceneText(11, 700)}>latency per token</text>
      {/* eval scorecard */}
      <rect x={358} y={56} width={86} height={94} rx={8} fill="#eef0ff" {...outlined} strokeWidth={2} />
      <text x={401} y={74} textAnchor="middle" {...sceneText(11, 700)}>evals</text>
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <rect x={368} y={84 + i * 16} width={12} height={10} rx={3} fill={i === 2 ? C.coral : C.mint} {...outlined} strokeWidth={2} />
          <line x1={386} y1={89 + i * 16} x2={434} y2={89 + i * 16} stroke={C.ink} strokeWidth={2} strokeLinecap="round" opacity={0.5} />
        </g>
      ))}
      {/* batch slots */}
      <text x={212} y={176} {...sceneText(11, 700)}>batch</text>
      {[0, 1, 2, 3, 4, 5].map((i) => <rect key={i} x={254 + i * 30} y={164} width={24} height={16} rx={4} fill={i < 4 ? C.amber : C.white} {...outlined} strokeWidth={2} strokeDasharray={i < 4 ? undefined : '4 3'} />)}
      {/* request queue */}
      {[0, 1, 2].map((i) => <Chip key={i} x={62 + i * 6} y={30 + i * 24} w={84} h={22} text={`request ${i + 1}`} mono={false} size={11} fill={C.white} rotate={-3 + i * 3} />)}
      <path d="M122 78 C 160 84, 170 118, 194 120" fill="none" {...outlined} strokeDasharray="2 8" />
      <Tok x={104} y={284} s={0.8} look={[4, -2]} mood="think" armR={[84, -92]} armL={[-58, -44]} />
      <Chip x={410} y={268} w={96} h={26} text="$ per 1M tokens" fill={C.mint} mono={false} size={11} />
      <Sparkle x={182} y={40} s={0.7} />
    </Scene>
  )
}

/* Part 11: graduation */
function Capstone() {
  const confetti: [number, number, string, number][] = [[60, 60, C.coral, 20], [120, 36, C.amber, -30], [200, 70, C.sky, 40], [400, 50, C.mint, 10], [440, 110, C.coral, -20], [350, 30, C.amber, 60], [90, 150, C.mint, 80], [420, 200, C.sky, -50]]
  return (
    <Scene label="Tok wears a graduation cap and holds up a tiny robot it built itself, with confetti falling">
      {confetti.map(([x, y, f, r], i) => <rect key={i} x={x} y={y} width={14} height={7} rx={2} fill={f} {...outlined} strokeWidth={2} transform={`rotate(${r} ${x} ${y})`} />)}
      <Tok
        x={216} y={276} look={[3, -3]} mood="happy" armR={[86, -150]} armL={[-72, -96]}
        hat={<g><path d="M-62 -182 L0 -204 L62 -182 L0 -160Z" fill={C.ink} {...outlined} /><path d="M40 -176 v26" {...outlined} /><circle cx={40} cy={-146} r={5} fill={C.amber} {...outlined} strokeWidth={2} /></g>}
      />
      {/* the mini robot */}
      <Tok x={316} y={118} s={0.34} mood="wow" />
      <Sparkle x={356} y={60} s={0.9} />
      <Sparkle x={276} y={44} s={0.6} fill={C.white} />
      <Chip x={96} y={236} w={120} h={28} text="built from scratch" fill={C.white} mono={false} size={11.5} rotate={-4} />
    </Scene>
  )
}

const SCENES: Record<string, () => ReactNode> = {
  hero: BigPicture,
  'big-picture': OpenBox,
  math: MathScene,
  learning: Learning,
  'neural-nets': NeuralNets,
  'text-to-numbers': TextToNumbers,
  'first-lm': FirstLm,
  attention: AttentionScene,
  gpt: Gpt,
  modern: Modern,
  systems: Systems,
  engineering: Engineering,
  capstone: Capstone,
}

export const PART_IDS_WITH_ART = Object.keys(SCENES)

/** The illustration for a part of the course. */
export function PartArt({ part }: { part: string }) {
  const S = SCENES[part] ?? BigPicture
  return <S />
}
