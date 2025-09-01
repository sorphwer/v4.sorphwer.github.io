'use strict'
exports.id = 642
exports.ids = [642]
exports.modules = {
  /***/ 642: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    // EXPORTS
    __webpack_require__.d(__webpack_exports__, {
      Z: () => /* binding */ generate_rss,
    }) // CONCATENATED MODULE: ./lib/utils/htmlEscaper.js

    const { replace } = ''
    // escape
    const es = /&(?:amp|#38|lt|#60|gt|#62|apos|#39|quot|#34);/g
    const ca = /[&<>'"]/g
    const esca = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;',
    }
    const pe = (m) => esca[m]
    /**
     * Safely escape HTML entities such as `&`, `<`, `>`, `"`, and `'`.
     * @param {string} es the input to safely escape
     * @returns {string} the escaped input, and it **throws** an error if
     *  the input type is unexpected, except for boolean and numbers,
     *  converted as string.
     */ const htmlEscaper_escape = (es1) => replace.call(es1, ca, pe)
    // EXTERNAL MODULE: ./data/siteMetadata.js
    var data_siteMetadata = __webpack_require__(1576) // CONCATENATED MODULE: ./lib/generate-rss.js
    const generateRssItem = (post) => `
  <item>
    <guid>${siteMetadata.siteUrl}/blog/${post.slug}</guid>
    <title>${escape(post.title)}</title>
    <link>${siteMetadata.siteUrl}/blog/${post.slug}</link>
    ${post.summary && `<description>${escape(post.summary)}</description>`}
    <pubDate>${new Date(post.date).toUTCString()}</pubDate>
    <author>${siteMetadata.email} (${siteMetadata.author})</author>
    ${post.tags && post.tags.map((t) => `<category>${t}</category>`).join('')}
  </item>
`
    // const generateRss = (posts, page = 'feed.xml') => `
    //   <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
    //     <channel>
    //       <title>${escape(siteMetadata.title)}</title>
    //       <link>${siteMetadata.siteUrl}/blog</link>
    //       <description>${escape(siteMetadata.description)}</description>
    //       <language>${siteMetadata.language}</language>
    //       <managingEditor>${siteMetadata.email} (${siteMetadata.author})</managingEditor>
    //       <webMaster>${siteMetadata.email} (${siteMetadata.author})</webMaster>
    //       <lastBuildDate>${new Date(posts[0].date).toUTCString()}</lastBuildDate>
    //       <atom:link href="${siteMetadata.siteUrl}/${page}" rel="self" type="application/rss+xml"/>
    //       ${posts.map(generateRssItem).join('')}
    //     </channel>
    //   </rss>
    // `
    const generateRss = (posts, page = 'feed.xml') => ''
    /* harmony default export */ const generate_rss = generateRss

    /***/
  },
}
