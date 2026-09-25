// Download GPT-2's weights once, keep them in the browser's Cache API, and report progress.
// A second visit reads the chunks from the cache: no network, no waiting.
// Runs in the Web Worker (gpt2.worker.ts); the Cache API exists there too (secure contexts: https or localhost).
import type { Gpt2Manifest } from './gpt2'

export const CACHE_NAME = 'llm-fp-gpt2-v1'

export interface LoadProgress { loaded: number; total: number; file: string; fromCache: boolean }
export interface Downloaded { man: Gpt2Manifest; chunks: ArrayBuffer[]; merges: ArrayBuffer; fromCache: boolean }

/** Cache key: the file URL plus its content hash, so re-exported weights never mix with stale ones. */
export const cacheKey = (baseUrl: string, file: string, sha: string) => new URL(`${baseUrl}${file}?v=${sha.slice(0, 12)}`, self.location.href).href

const openCache = async (): Promise<Cache | null> => {
  try {
    return typeof caches === 'undefined' ? null : await caches.open(CACHE_NAME)
  } catch {
    return null // private mode or an insecure origin: download every time
  }
}

/** Stream one file into a buffer of known size, calling onBytes as data arrives. */
const fetchInto = async (url: string, size: number, onBytes: (n: number) => void, signal?: AbortSignal): Promise<ArrayBuffer> => {
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`could not download ${url.split('/').pop()} (HTTP ${res.status})`)
  if (!res.body) {
    const b = await res.arrayBuffer()
    onBytes(b.byteLength)
    return b
  }
  const out = new Uint8Array(size)
  const reader = res.body.getReader()
  let o = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (o + value.length > size) throw new Error(`${url.split('/').pop()} is larger than the manifest says`)
    out.set(value, o)
    o += value.length
    onBytes(value.length)
  }
  if (o !== size) throw new Error(`${url.split('/').pop()} arrived incomplete (${o} of ${size} bytes)`)
  return out.buffer
}

/** Which files of this manifest are already cached. */
export const cachedFiles = async (baseUrl: string, man: Gpt2Manifest): Promise<number> => {
  const cache = await openCache()
  if (!cache) return 0
  let n = 0
  for (const c of man.chunks) if (await cache.match(cacheKey(baseUrl, c.file, c.sha256))) n++
  return n
}

export const fetchManifest = async (baseUrl: string): Promise<Gpt2Manifest> => {
  const res = await fetch(`${baseUrl}manifest.json`, { cache: 'no-cache' })
  if (!res.ok) throw new Error(`could not load the GPT-2 manifest (HTTP ${res.status})`)
  return (await res.json()) as Gpt2Manifest
}

export const downloadGpt2 = async (baseUrl: string, onProgress: (p: LoadProgress) => void, signal?: AbortSignal): Promise<Downloaded> => {
  const man = await fetchManifest(baseUrl)
  const cache = await openCache()
  const files = [...man.chunks, { file: man.tokenizer.file, bytes: man.tokenizer.bytes, sha256: `merges${man.tokenizer.n_merges}` }]
  const total = files.reduce((a, f) => a + f.bytes, 0)
  let loaded = 0
  let allCached = true
  const bufs: ArrayBuffer[] = []
  for (const f of files) {
    const key = cacheKey(baseUrl, f.file, f.sha256)
    let buf: ArrayBuffer | null = null
    try {
      const hit = cache ? await cache.match(key) : undefined
      if (hit) {
        buf = await hit.arrayBuffer()
        if (buf.byteLength !== f.bytes) buf = null // damaged entry: fetch again
      }
    } catch {
      buf = null
    }
    if (buf) {
      loaded += f.bytes
      onProgress({ loaded, total, file: f.file, fromCache: true })
    } else {
      allCached = false
      buf = await fetchInto(`${baseUrl}${f.file}`, f.bytes, (n) => { loaded += n; onProgress({ loaded, total, file: f.file, fromCache: false }) }, signal)
      try {
        await cache?.put(key, new Response(buf.slice(0), { headers: { 'Content-Type': 'application/octet-stream' } }))
      } catch {
        /* storage full or refused: the model still works, it just downloads again next time */
      }
    }
    bufs.push(buf)
  }
  // remove entries left behind by an older export
  if (cache) {
    try {
      const keep = new Set(files.map((f) => cacheKey(baseUrl, f.file, f.sha256)))
      for (const req of await cache.keys()) if (!keep.has(req.url)) await cache.delete(req)
    } catch { /* ignore */ }
  }
  return { man, chunks: bufs.slice(0, man.chunks.length), merges: bufs[bufs.length - 1], fromCache: allCached }
}
