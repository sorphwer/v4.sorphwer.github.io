// Minimal stateless MCP server (Streamable HTTP transport, JSON responses only).
// Exposed at /mcp via a rewrite in next.config.js. Its single tool echoes back
// the HTTP request headers of the call, for testing what an MCP client sends.

const SUPPORTED_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05']
const SERVER_INFO = { name: 'riino-header-echo', version: '1.0.0' }

const TOOL = {
  name: 'get_request_headers',
  title: 'Get request headers',
  description:
    'Returns every HTTP header of the request that called this tool, as received by the server.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
}

// The hosting platform injects its own credentials into every request it
// forwards (Vercel's OIDC token, runtime-cache bearer, proxy signatures).
// They are not part of what the client sent and must never be echoed.
const HIDDEN_HEADER =
  /^(x-vercel-oidc-token|x-vercel-sc-|x-vercel-proxy-signature|x-vercel-internal-)/i
const visible = (name) => !HIDDEN_HEADER.test(name)
const scrub = (name, value) =>
  name.toLowerCase() === 'forwarded' ? String(value).replace(/;?\s*sig=[^;,]*/gi, '') : value

function headerSnapshot(req) {
  const headers = {}
  for (const [name, value] of Object.entries(req.headers)) {
    if (visible(name)) headers[name] = scrub(name, value)
  }
  const raw = []
  for (let i = 0; i < req.rawHeaders.length; i += 2) {
    const name = req.rawHeaders[i]
    if (visible(name)) raw.push([name, scrub(name, req.rawHeaders[i + 1])])
  }
  return { method: req.method, url: req.url, headers, rawHeaders: raw }
}

const ok = (id, result) => ({ jsonrpc: '2.0', id, result })
const fail = (id, code, message) => ({ jsonrpc: '2.0', id, error: { code, message } })

function handleMessage(msg, req) {
  if (!msg || msg.jsonrpc !== '2.0' || typeof msg.method !== 'string') {
    return fail(msg?.id ?? null, -32600, 'Invalid Request')
  }
  const isNotification = msg.id === undefined || msg.id === null
  if (isNotification) return null

  const { id, method, params } = msg
  switch (method) {
    case 'initialize': {
      const asked = params?.protocolVersion
      return ok(id, {
        protocolVersion: SUPPORTED_VERSIONS.includes(asked) ? asked : SUPPORTED_VERSIONS[0],
        capabilities: { tools: { listChanged: false } },
        serverInfo: SERVER_INFO,
        instructions: 'Call get_request_headers to see the HTTP headers your client sent.',
      })
    }
    case 'ping':
      return ok(id, {})
    case 'tools/list':
      return ok(id, { tools: [TOOL] })
    case 'tools/call': {
      if (params?.name !== TOOL.name) return fail(id, -32602, `Unknown tool: ${params?.name}`)
      const snapshot = headerSnapshot(req)
      return ok(id, {
        content: [{ type: 'text', text: JSON.stringify(snapshot, null, 2) }],
        structuredContent: snapshot,
        isError: false,
      })
    }
    default:
      return fail(id, -32601, `Method not found: ${method}`)
  }
}

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, DELETE, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', '*')
  res.setHeader('Access-Control-Expose-Headers', 'Mcp-Session-Id, Mcp-Protocol-Version')
  res.setHeader('Cache-Control', 'no-store')

  if (req.method === 'OPTIONS') return res.status(204).end()
  if (req.method !== 'POST') {
    // Stateless server: no SSE stream (GET) and no sessions to end (DELETE).
    res.setHeader('Allow', 'POST, OPTIONS')
    return res.status(405).json(fail(null, -32000, 'Method not allowed'))
  }

  let body = req.body
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body)
    } catch {
      return res.status(400).json(fail(null, -32700, 'Parse error'))
    }
  }

  if (Array.isArray(body)) {
    const out = body.map((m) => handleMessage(m, req)).filter(Boolean)
    return out.length ? res.status(200).json(out) : res.status(202).end()
  }
  const out = handleMessage(body, req)
  return out ? res.status(200).json(out) : res.status(202).end()
}
