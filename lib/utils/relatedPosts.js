// Source-kind tags added by getAllFilesFrontMatter; every post has one, so they
// say nothing about topic.
const SOURCE_TAGS = new Set(['mdx', 'notion'])

const topicTags = (post) =>
  (post.tags ?? []).filter((tag) => !SOURCE_TAGS.has(String(tag).toLowerCase()))

// Props must be JSON: some old posts have no title (fall back to the slug).
const card = (post, kind, label) => ({
  slug: post.slug,
  title: post.title || post.slug,
  date: post.date ?? null,
  kind,
  label,
})

/**
 * Cards for the "Keep reading" section: the previous (older) and next (newer)
 * posts, then the posts sharing the most topic tags with `post` (ties go to the
 * nearest date), up to `limit` cards in total.
 */
export default function relatedPosts(allPosts, post, prev, next, limit = 4) {
  const cards = []
  if (prev) cards.push(card(prev, 'prev', 'Previous'))
  if (next) cards.push(card(next, 'next', 'Next'))

  const tags = new Set(topicTags(post))
  const taken = new Set([post.slug, prev?.slug, next?.slug])
  const time = new Date(post.date).getTime()
  const ranked = allPosts
    .filter((other) => !taken.has(other.slug))
    .map((other) => {
      const shared = topicTags(other).filter((tag) => tags.has(tag))
      return {
        other,
        shared,
        distance: Math.abs(new Date(other.date).getTime() - time),
      }
    })
    .filter(({ shared }) => shared.length > 0)
    .sort((a, b) => b.shared.length - a.shared.length || a.distance - b.distance)

  for (const { other, shared } of ranked) {
    if (cards.length >= limit) break
    cards.push(card(other, 'tag', shared[0]))
  }
  return cards
}
