// Fast source preview for prose and layout; Julia/Documenter remains the full build.
import { readFileSync, writeFileSync, mkdirSync, cpSync, watch, readdirSync, existsSync, rmSync, statSync } from 'node:fs'
import { dirname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vitepress'

const docs = dirname(fileURLToPath(import.meta.url))
const source = join(docs, 'src')
const output = join(docs, 'build', '.source-preview')
const makefile = join(docs, 'make.jl')

// Read the string pairs and nested groups in makedocs(pages=[...]). Keeping
// this list in make.jl gives the source preview and published docs the same TOC.
function sidebar() {
  const make = readFileSync(makefile, 'utf8')
  const start = make.indexOf('pages=[')
  if (start < 0) throw new Error('Cannot find pages=[...] in docs/make.jl')
  const text = make.slice(start + 'pages='.length)
  let pos = 0
  function token() {
    while (/\s/.test(text[pos] ?? '') && pos < text.length) pos++
    const match = /^("(?:\\.|[^"\\])*"|=>|\[|\]|,)/.exec(text.slice(pos))
    if (!match) throw new Error(`Unsupported docs navigation near ${text.slice(pos, pos + 60)}`)
    pos += match[0].length
    return match[0]
  }
  function group() {
    const items = []
    let next = token()
    while (next !== ']') {
      const label = JSON.parse(next)
      if (token() !== '=>') throw new Error('Expected a docs page pair')
      const value = token()
      items.push(value === '['
        ? { text: label, collapsed: true, items: group() }
        : { text: label, link: '/' + JSON.parse(value).replace(/\.md$/, '') })
      next = token()
      if (next === ',') next = token()
    }
    return items
  }
  if (token() !== '[') throw new Error('Expected docs pages array')
  return group()
}

function markdown(text) {
  // Preserve code examples without executing Julia. Clearly mark content that
  // only Documenter can expand instead of presenting it as rendered API docs.
  return text.replace(/^(`{3,})(@\w+)([^\n]*)\n([\s\S]*?)^\1[ \t]*$/gm,
    (_, fence, kind, args, body) => {
      if (kind === '@raw' && args.trim() === 'html') return body
      if (kind === '@meta' || kind === '@setup') return ''
      if (kind === '@docs' || kind === '@autodocs') {
        return '::: info API documentation (full build only)\nThe published page includes API docstrings here. They are omitted from this local source preview.\n:::\n'
      }
      if (kind === '@index' || kind === '@contents') {
        return '::: info Navigation (full build only)\nThe full Julia documentation build generates this navigation block.\n:::\n'
      }
      return `\`\`\`julia\n${body}\`\`\`\n`
    })
    .replace(/^```math\s*\n([\s\S]*?)^```\s*$/gm, (_, body) => `$$\n${body}$$\n`)
    .replace(/(?<!`)``([^`\n]+)``(?!`)/g, (_, math) => `$${math}$`)
    .replace(/(\]\([^\s)]*\.md)#([^\s)]+)/g, (_, page, anchor) =>
      `${page}#${decodeURIComponent(anchor).toLowerCase().replace(/[^\p{L}\p{N}_-]/gu, '')}`)
}

function config() {
  const replacements = {
    base: '/', title: 'cuNumeric.jl', description: 'Distributed arrays in Julia',
    outDir: '../preview-dist', nav: [], sidebar: sidebar(),
    editLink: { pattern: 'https://github.com/JuliaLegate/cuNumeric.jl/edit/main/docs/src/:path' },
  }
  let text = readFileSync(join(source, '.vitepress', 'config.mts'), 'utf8')
  for (const [key, value] of Object.entries(replacements)) {
    text = text.replace(new RegExp(`\\b${key}: 'REPLACE_ME_DOCUMENTER_VITEPRESS'`, 'g'), `${key}: ${JSON.stringify(value)}`)
  }
  return text
    .replaceAll('REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON', '/assets/logo.png')
    .replaceAll('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH', '/')
    .replaceAll('REPLACE_ME_DOCUMENTER_VITEPRESS', 'https://github.com/JuliaLegate/cuNumeric.jl')
    // Published version metadata is not needed by a local source preview.
    .replace(/^\s*\['script',.*$/gm, '')
    .replace('    nav,', '    nav: [],')
}

function syncFile(relative) {
  if (!relative || /(^|[\\/])(cache|dist|node_modules)([\\/]|$)/.test(relative)) return
  const target = resolve(output, relative)
  if (!target.startsWith(output + sep)) throw new Error('Preview path escaped its output directory')
  if (!existsSync(join(source, relative))) {
    rmSync(target, { recursive: true, force: true })
    return
  }
  if (statSync(join(source, relative)).isDirectory()) {
    mkdirSync(target, { recursive: true })
    for (const entry of readdirSync(join(source, relative))) {
      syncFile(join(relative, entry))
    }
    return
  }
  mkdirSync(dirname(target), { recursive: true })
  if (relative.replaceAll('\\', '/') === '.vitepress/config.mts') {
    const nextConfig = config()
    if (!existsSync(target) || readFileSync(target, 'utf8') !== nextConfig) {
      writeFileSync(target, nextConfig)
    }
  } else if (relative.endsWith('.md')) {
    writeFileSync(target, markdown(readFileSync(join(source, relative), 'utf8')))
  } else {
    cpSync(join(source, relative), target, { recursive: true })
  }
}

mkdirSync(output, { recursive: true })
cpSync(source, output, {
  recursive: true,
  filter: file => !/(^|[\\/])(cache|dist|node_modules)([\\/]|$)/.test(file),
})
function syncMarkdown(relative = '') {
  for (const entry of readdirSync(join(source, relative), { withFileTypes: true })) {
    if (['cache', 'dist', 'node_modules'].includes(entry.name)) continue
    const file = join(relative, entry.name)
    if (entry.isDirectory()) syncMarkdown(file)
    else if (entry.name.endsWith('.md')) syncFile(file)
  }
}
syncMarkdown()
syncFile('.vitepress/config.mts')

const timers = new Map()
const sourceWatcher = watch(source, { recursive: true }, (_, file) => {
  if (!file) return
  clearTimeout(timers.get(file))
  timers.set(file, setTimeout(() => {
    timers.delete(file)
    try { syncFile(file) } catch (error) { console.error(`Preview: ${file}: ${error.message}`) }
  }, 100))
})
const navWatcher = watch(makefile, () => syncFile('.vitepress/config.mts'))
let server
let restartPromise
async function startServer(isRestart = false) {
  const current = await createServer(resolve(output),
    { host: '127.0.0.1', port: 5173, strictPort: true },
    () => {
      if (!restartPromise) {
        restartPromise = (async () => {
          await current.close()
          await startServer(true)
        })().finally(() => { restartPromise = undefined })
      }
      return restartPromise
    })
  server = current
  await current.listen(undefined, isRestart)
  current.printUrls()
}
await startServer()
console.log('Watching docs/src and docs/make.jl. Julia examples and API directives require the full docs build.')
async function stop() {
  sourceWatcher.close()
  navWatcher.close()
  for (const timer of timers.values()) clearTimeout(timer)
  await server.close()
  process.exit(0)
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
