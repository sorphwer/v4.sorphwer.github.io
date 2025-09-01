'use strict'
;(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([
  [673],
  {
    9831: function (t, e, r) {
      r.d(e, {
        TQ: function () {
          return s
        },
        $t: function () {
          return d
        },
        Uy: function () {
          return m
        },
      })
      var a = r(7320),
        n = r(9008),
        i = r(1163),
        o = r(1576),
        l = r.n(o),
        c = function (t) {
          var e = t.title,
            r = t.description,
            o = t.ogType,
            c = t.ogImage,
            s = t.twImage,
            d = t.canonicalUrl,
            m = (0, i.useRouter)()
          return (0, a.BX)(n.default, {
            children: [
              (0, a.tZ)('title', { children: e }),
              (0, a.tZ)('meta', { name: 'robots', content: 'follow, index' }),
              (0, a.tZ)('meta', { name: 'description', content: r }),
              (0, a.tZ)('meta', {
                property: 'og:url',
                content: ''.concat(l().siteUrl).concat(m.asPath),
              }),
              (0, a.tZ)('meta', { property: 'og:type', content: o }),
              (0, a.tZ)('meta', { property: 'og:site_name', content: l().title }),
              (0, a.tZ)('meta', { property: 'og:description', content: r }),
              (0, a.tZ)('meta', { property: 'og:title', content: e }),
              'Array' === c.constructor.name
                ? c.map(function (t) {
                    var e = t.url
                    return (0, a.tZ)('meta', { property: 'og:image', content: e }, e)
                  })
                : (0, a.tZ)('meta', { property: 'og:image', content: c }, c),
              (0, a.tZ)('meta', { name: 'twitter:card', content: 'summary_large_image' }),
              (0, a.tZ)('meta', { name: 'twitter:site', content: l().twitter }),
              (0, a.tZ)('meta', { name: 'twitter:title', content: e }),
              (0, a.tZ)('meta', { name: 'twitter:description', content: r }),
              (0, a.tZ)('meta', { name: 'twitter:image', content: s }),
              (0, a.tZ)('link', {
                rel: 'canonical',
                href: d || ''.concat(l().siteUrl).concat(m.asPath),
              }),
            ],
          })
        },
        s = function (t) {
          var e = t.title,
            r = t.description,
            n = l().siteUrl + l().socialBanner,
            i = l().siteUrl + l().socialBanner
          return (0, a.tZ)(c, {
            title: e,
            description: r,
            ogType: 'website',
            ogImage: n,
            twImage: i,
          })
        },
        d = function (t) {
          var e = t.title,
            r = t.description,
            o = l().siteUrl + l().socialBanner,
            s = l().siteUrl + l().socialBanner,
            d = (0, i.useRouter)()
          return (0, a.BX)(a.HY, {
            children: [
              (0, a.tZ)(c, { title: e, description: r, ogType: 'website', ogImage: o, twImage: s }),
              (0, a.tZ)(n.default, {
                children: (0, a.tZ)('link', {
                  rel: 'alternate',
                  type: 'application/rss+xml',
                  title: ''.concat(r, ' - RSS feed'),
                  href: ''.concat(l().siteUrl).concat(d.asPath, '/feed.xml'),
                }),
              }),
            ],
          })
        },
        m = function (t) {
          var e = t.authorDetails,
            r = t.title,
            o = t.summary,
            s = t.date,
            d = t.lastmod,
            m = t.url,
            u = t.images,
            g = void 0 === u ? [] : u,
            p = t.canonicalUrl,
            h = ((0, i.useRouter)(), new Date(s).toISOString()),
            y = new Date(d || s).toISOString(),
            f = (0 === g.length ? [l().socialBanner] : 'string' === typeof g ? [g] : g).map(
              function (t) {
                return { '@type': 'ImageObject', url: t.includes('http') ? t : l().siteUrl + t }
              }
            ),
            b = {
              '@context': 'https://schema.org',
              '@type': 'Article',
              mainEntityOfPage: { '@type': 'WebPage', '@id': m },
              headline: r,
              image: f,
              datePublished: h,
              dateModified: y,
              author: e
                ? e.map(function (t) {
                    return { '@type': 'Person', name: t.name }
                  })
                : { '@type': 'Person', name: l().author },
              publisher: {
                '@type': 'Organization',
                name: l().author,
                logo: { '@type': 'ImageObject', url: ''.concat(l().siteUrl).concat(l().siteLogo) },
              },
              description: o,
            },
            x = f[0].url
          return (0, a.BX)(a.HY, {
            children: [
              (0, a.tZ)(c, {
                title: r,
                description: o,
                ogType: 'article',
                ogImage: f,
                twImage: x,
                canonicalUrl: p,
              }),
              (0, a.BX)(n.default, {
                children: [
                  s && (0, a.tZ)('meta', { property: 'article:published_time', content: h }),
                  d && (0, a.tZ)('meta', { property: 'article:modified_time', content: y }),
                  (0, a.tZ)('script', {
                    type: 'application/ld+json',
                    dangerouslySetInnerHTML: { __html: JSON.stringify(b, null, 2) },
                  }),
                ],
              }),
            ],
          })
        }
    },
    9019: function (t, e, r) {
      var a = r(7320),
        n = r(1664),
        i = r(4871),
        o = r(7814),
        l = r(5675)
      e.Z = function (t) {
        var e = t.text
        return 'notion' === (0, i.Z)(e)
          ? (0, a.tZ)(n.default, {
              href: '/tags/'.concat((0, i.Z)(e)),
              children: (0, a.BX)('a', {
                className:
                  'mt-1 mr-3 rounded border-2 border-solid border-black bg-violet-700 px-2 text-sm font-medium uppercase text-black transition duration-500 ease-out hover:border-primary-500 hover:bg-gray-300 hover:text-primary-500 dark:border-gray-300 dark:text-gray-300 dark:hover:border-primary-400 hover:dark:bg-gray-500 dark:hover:text-primary-400',
                children: [
                  (0, a.tZ)(l.default, {
                    className: 'brightness-0 filter dark:brightness-200 dark:filter',
                    src: '/static/images/notion.svg',
                    width: 14,
                    height: 14,
                    alt: 'Notion Blog',
                  }),
                  ' ' + e.split(' ').join('-'),
                ],
              }),
            })
          : 'mdx' === (0, i.Z)(e)
          ? (0, a.tZ)(n.default, {
              href: '/tags/'.concat((0, i.Z)(e)),
              children: (0, a.tZ)('a', {
                className:
                  'mt-1 mr-3 rounded border-2 border-solid border-black bg-white p-0 text-sm font-medium uppercase text-black transition duration-500 ease-out hover:border-primary-500 hover:text-primary-500 dark:border-gray-300 dark:text-gray-300 dark:hover:border-primary-400 dark:hover:text-primary-400',
                children: (0, a.tZ)(l.default, {
                  src: '/static/images/mdx.png',
                  width: 34,
                  height: 14,
                  alt: 'mdx',
                }),
              }),
            })
          : (0, a.tZ)(n.default, {
              href: '/tags/'.concat((0, i.Z)(e)),
              children: (0, a.BX)('a', {
                className:
                  'mt-1 mr-3 rounded border-2 border-solid border-black px-2 text-sm font-medium uppercase text-black transition duration-500 ease-out hover:border-primary-500 hover:bg-gray-300 hover:text-primary-500 dark:border-gray-300 dark:text-gray-300 dark:hover:border-primary-400 hover:dark:bg-gray-500 dark:hover:text-primary-400',
                children: [
                  (0, a.tZ)(o.G, { icon: 'tags', className: 'text-black dark:text-gray-300 ' }),
                  ' ' + e.split(' ').join('-'),
                ],
              }),
            })
      }
    },
    6055: function (t, e, r) {
      r.r(e),
        r.d(e, {
          default: function () {
            return d
          },
        })
      var a = r(7320),
        n = r(7233),
        i = r(9019),
        o = (r(5675), r(1576), r(1720))
      function l(t) {
        var e = t.totalPages,
          r = t.currentPage,
          i = parseInt(r) - 1 > 0,
          o = parseInt(r) + 1 <= parseInt(e)
        return (0, a.tZ)('div', {
          className: 'space-y-2 pt-6 pb-8 md:space-y-5',
          children: (0, a.BX)('nav', {
            className: 'flex justify-between',
            children: [
              !i &&
                (0, a.tZ)('button', {
                  rel: 'previous',
                  className: 'cursor-auto disabled:opacity-50',
                  disabled: !i,
                  children: 'Previous',
                }),
              i &&
                (0, a.tZ)(n.Z, {
                  href: r - 1 === 1 ? '/blog/' : '/blog/page/'.concat(r - 1),
                  children: (0, a.tZ)('button', { rel: 'previous', children: 'Previous' }),
                }),
              (0, a.BX)('span', { children: [r, ' of ', e] }),
              !o &&
                (0, a.tZ)('button', {
                  rel: 'next',
                  className: 'cursor-auto disabled:opacity-50',
                  disabled: !o,
                  children: 'Next',
                }),
              o &&
                (0, a.tZ)(n.Z, {
                  href: '/blog/page/'.concat(r + 1),
                  children: (0, a.tZ)('button', { rel: 'next', children: 'Next' }),
                }),
            ],
          }),
        })
      }
      var c = r(6232),
        s = r(7814)
      r(9159)
      function d(t) {
        var e = t.posts,
          r = (t.title, t.initialDisplayPosts),
          d = void 0 === r ? [] : r,
          m = t.pagination,
          u = t.enableSearch,
          g = void 0 === u || u,
          p = (0, o.useState)(''),
          h = p[0],
          y = p[1],
          f = e.filter(function (t) {
            return (t.title + t.tags.join(' ')).toLowerCase().includes(h.toLowerCase())
          }),
          b = d.length > 0 && !h ? d : f
        return (0, a.BX)(a.HY, {
          children: [
            (0, a.BX)('div', {
              className: 'divide-y divide-gray-200 dark:divide-gray-700',
              children: [
                (0, a.tZ)('div', {
                  className: 'space-y-2 pt-6 pb-8 md:space-y-5',
                  children:
                    g &&
                    (0, a.BX)('div', {
                      className: ' relative max-w-4xl ',
                      children: [
                        (0, a.tZ)('input', {
                          'aria-label': 'Search articles',
                          type: 'text',
                          onChange: function (t) {
                            return y(t.target.value)
                          },
                          placeholder: 'Search articles',
                          className:
                            'block w-full rounded-md border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-primary-500 focus:ring-primary-500 dark:border-gray-900 dark:bg-gray-800 dark:text-gray-100',
                        }),
                        (0, a.tZ)('svg', {
                          className:
                            'absolute right-3 top-3 h-5 w-5 text-gray-400 dark:text-gray-300',
                          xmlns: 'http://www.w3.org/2000/svg',
                          fill: 'none',
                          viewBox: '0 0 24 24',
                          stroke: 'currentColor',
                          children: (0, a.tZ)('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
                          }),
                        }),
                      ],
                    }),
                }),
                (0, a.BX)('ul', {
                  children: [
                    !f.length && 'No posts found.',
                    b.map(function (t) {
                      var e = t.slug,
                        r = t.date,
                        o = t.title,
                        l = t.summary,
                        d = t.tags,
                        m = Math.floor((new Date(r).getMonth() / 12) * 4) % 4
                      return (0, a.tZ)(
                        'li',
                        {
                          className: 'py-4 font-rs',
                          children: (0, a.BX)('article', {
                            className:
                              'space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0',
                            children: [
                              (0, a.BX)('dl', {
                                children: [
                                  (0, a.tZ)('dt', {
                                    className: 'sr-only',
                                    children: 'Published on',
                                  }),
                                  (0, a.BX)('dd', {
                                    className:
                                      'text -base font-medium leading-6 text-gray-500 dark:text-gray-400',
                                    children: [
                                      0 == m &&
                                        (0, a.tZ)(s.G, { icon: 'fan', className: 'text-pink-300' }),
                                      1 == m &&
                                        (0, a.tZ)(s.G, {
                                          icon: 'sun',
                                          className: 'text-amber-300',
                                        }),
                                      2 == m &&
                                        (0, a.tZ)(s.G, {
                                          icon: 'leaf',
                                          className: 'text-green-300',
                                        }),
                                      3 == m &&
                                        (0, a.tZ)(s.G, {
                                          icon: 'snowflake',
                                          className: 'text-stone-300',
                                        }),
                                      (0, a.tZ)('time', {
                                        dateTime: r,
                                        children: ' ' + (0, c.Z)(r),
                                      }),
                                    ],
                                  }),
                                ],
                              }),
                              (0, a.BX)('div', {
                                className: 'space-y-3 font-rs xl:col-span-3',
                                children: [
                                  (0, a.tZ)('div', {
                                    children: (0, a.BX)('div', {
                                      children: [
                                        (0, a.BX)('h3', {
                                          className: 'text-2xl font-bold leading-8 tracking-tight ',
                                          children: [
                                            (0, a.tZ)(n.Z, {
                                              href: '/blog/'.concat(e),
                                              className:
                                                'text-gray-900 hover:text-primary-600 dark:text-gray-100 dark:hover:text-primary-400',
                                              children: o || e,
                                            }),
                                            t.status &&
                                              (0, a.tZ)('span', {
                                                className:
                                                  'align-top text-sm font-normal text-RSpink',
                                                children: ' [' + t.status + ']',
                                              }),
                                          ],
                                        }),
                                        t.subtitle &&
                                          (0, a.tZ)('div', {
                                            className: 'text-xl font-normal text-gray-500',
                                            children: t.subtitle,
                                          }),
                                        (0, a.tZ)('div', {
                                          className: 'flex flex-wrap ',
                                          children: d.map(function (t) {
                                            return (0, a.tZ)(i.Z, { text: t }, t)
                                          }),
                                        }),
                                      ],
                                    }),
                                  }),
                                  (0, a.tZ)('div', {
                                    className: 'prose max-w-none text-gray-500 dark:text-gray-400',
                                    children: l,
                                  }),
                                ],
                              }),
                            ],
                          }),
                        },
                        e
                      )
                    }),
                  ],
                }),
              ],
            }),
            m &&
              m.totalPages > 1 &&
              !h &&
              (0, a.tZ)(l, { currentPage: m.currentPage, totalPages: m.totalPages }),
          ],
        })
      }
    },
    6232: function (t, e, r) {
      var a = r(1576),
        n = r.n(a)
      e.Z = function (t) {
        return new Date(t).toLocaleDateString(n().locale, {
          year: 'numeric',
          month: 'long',
          day: 'numeric',
        })
      }
    },
    4871: function (t, e, r) {
      var a = r(9671)
      e.Z = function (t) {
        return (0, a.slug)(t)
      }
    },
  },
])
