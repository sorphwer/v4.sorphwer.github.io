import { NotionAPI } from 'notion-client'

// Notion's unofficial API (www.notion.so/api/v3) sits behind Cloudflare, which
// returns 403 to got's default User-Agent. A browser UA is enough to pass.
const BROWSER_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'

const notion = new NotionAPI()

// Notion now returns records as `{ value: { value: {...}, role } }` instead of
// `{ value: {...}, role }`. notion-client@6 / react-notion-x@6 expect the old
// shape (they read `block.value.id`), so unwrap the extra layer.
function unwrapRecords(records) {
  if (!records) return records
  for (const record of Object.values(records)) {
    const inner = record?.value?.value
    if (inner && typeof inner === 'object' && 'id' in inner) {
      record.role = record.role ?? record.value.role
      record.value = inner
    }
  }
  return records
}

export async function getNotionPage(pageId) {
  const recordMap = await notion.getPage(pageId, {
    gotOptions: { headers: { 'user-agent': BROWSER_UA } },
  })
  unwrapRecords(recordMap.block)
  unwrapRecords(recordMap.collection)
  unwrapRecords(recordMap.collection_view)
  unwrapRecords(recordMap.notion_user)
  return recordMap
}
