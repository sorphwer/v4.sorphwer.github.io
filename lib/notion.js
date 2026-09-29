import { NotionAPI } from 'notion-client'

// Notion's unofficial API (www.notion.so/api/v3) sits behind Cloudflare, which
// returns 403 to got's default User-Agent. A browser UA is enough to pass.
const BROWSER_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'

// Notion now returns records as `{ value: { value: {...}, role } }` instead of
// `{ value: {...}, role }` (and inaccessible ones as `{ value: { role: 'none' } }`
// instead of `{ role: 'none' }`). notion-client@6 / react-notion-x@6 expect the
// old shape (they read `record.value.id` / `.type`), so unwrap the extra layer.
function unwrapRecords(records) {
  if (!records || typeof records !== 'object') return
  for (const record of Object.values(records)) {
    const value = record?.value
    if (!value || typeof value !== 'object' || 'id' in value) continue
    record.role = record.role ?? value.role
    if (value.value && typeof value.value === 'object') record.value = value.value
    else delete record.value
  }
}

class CompatNotionAPI extends NotionAPI {
  // Every API call (loadPageChunk, syncRecordValues, queryCollection,
  // getSignedFileUrls) goes through fetch(). Unwrapping here means getPage's
  // own post-processing (fetching collections, signing file URLs) sees the
  // shape it expects.
  async fetch({ headers, ...rest }) {
    const res = await super.fetch({
      ...rest,
      headers: { 'user-agent': BROWSER_UA, ...headers },
    })
    const recordMap = res?.recordMap
    if (recordMap) {
      for (const table of Object.values(recordMap)) unwrapRecords(table)
    }
    return res
  }
}

const notion = new CompatNotionAPI()

export function getNotionPage(pageId) {
  return notion.getPage(pageId)
}
