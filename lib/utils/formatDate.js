import siteMetadata from '@/data/siteMetadata'

const LONG_DATE = { year: 'numeric', month: 'long', day: 'numeric' }
/** "Sep 24, 2026", for cards. */
export const SHORT_DATE = { year: 'numeric', month: 'short', day: 'numeric' }

/** `date` in the site locale; long form ("September 24, 2026") unless `options` says otherwise. */
const formatDate = (date, options = LONG_DATE) =>
  new Date(date).toLocaleDateString(siteMetadata.locale, options)

export default formatDate
