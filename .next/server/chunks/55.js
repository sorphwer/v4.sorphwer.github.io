'use strict'
exports.id = 55
exports.ids = [55]
exports.modules = {
  /***/ 6055: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    // ESM COMPAT FLAG
    __webpack_require__.r(__webpack_exports__)

    // EXPORTS
    __webpack_require__.d(__webpack_exports__, {
      default: () => /* binding */ ListLayout,
    })

    // EXTERNAL MODULE: external "react/jsx-runtime"
    var jsx_runtime_ = __webpack_require__(997)
    // EXTERNAL MODULE: ./components/Link.js
    var Link = __webpack_require__(7233)
    // EXTERNAL MODULE: ./components/Tag.js
    var Tag = __webpack_require__(9019)
    // EXTERNAL MODULE: ./node_modules/next/image.js
    var next_image = __webpack_require__(5675)
    // EXTERNAL MODULE: ./data/siteMetadata.js
    var siteMetadata = __webpack_require__(1576)
    // EXTERNAL MODULE: external "react"
    var external_react_ = __webpack_require__(6689) // CONCATENATED MODULE: ./components/Pagination.js
    function Pagination({ totalPages, currentPage }) {
      const prevPage = parseInt(currentPage) - 1 > 0
      const nextPage = parseInt(currentPage) + 1 <= parseInt(totalPages)
      return /*#__PURE__*/ jsx_runtime_.jsx('div', {
        className: 'space-y-2 pt-6 pb-8 md:space-y-5',
        children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('nav', {
          className: 'flex justify-between',
          children: [
            !prevPage &&
              /*#__PURE__*/ jsx_runtime_.jsx('button', {
                rel: 'previous',
                className: 'cursor-auto disabled:opacity-50',
                disabled: !prevPage,
                children: 'Previous',
              }),
            prevPage &&
              /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                href: currentPage - 1 === 1 ? `/blog/` : `/blog/page/${currentPage - 1}`,
                children: /*#__PURE__*/ jsx_runtime_.jsx('button', {
                  rel: 'previous',
                  children: 'Previous',
                }),
              }),
            /*#__PURE__*/ (0, jsx_runtime_.jsxs)('span', {
              children: [currentPage, ' of ', totalPages],
            }),
            !nextPage &&
              /*#__PURE__*/ jsx_runtime_.jsx('button', {
                rel: 'next',
                className: 'cursor-auto disabled:opacity-50',
                disabled: !nextPage,
                children: 'Next',
              }),
            nextPage &&
              /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                href: `/blog/page/${currentPage + 1}`,
                children: /*#__PURE__*/ jsx_runtime_.jsx('button', {
                  rel: 'next',
                  children: 'Next',
                }),
              }),
          ],
        }),
      })
    }

    // EXTERNAL MODULE: ./lib/utils/formatDate.js
    var formatDate = __webpack_require__(6232)
    // EXTERNAL MODULE: external "@fortawesome/react-fontawesome"
    var react_fontawesome_ = __webpack_require__(7197)
    // EXTERNAL MODULE: ./components/social-icons/index.js + 7 modules
    var social_icons = __webpack_require__(9159) // CONCATENATED MODULE: ./layouts/ListLayout.js
    // Archive page

    function ListLayout({
      posts,
      title: title1,
      initialDisplayPosts = [],
      pagination,
      enableSearch = true,
    }) {
      const { 0: searchValue, 1: setSearchValue } = (0, external_react_.useState)('')
      // posts is the all frontmatter
      // console.log(posts)
      const filteredBlogPosts = posts.filter((frontMatter) => {
        const searchContent = frontMatter.title + frontMatter.tags.join(' ')
        return searchContent.toLowerCase().includes(searchValue.toLowerCase())
      })
      // If initialDisplayPosts exist, display it if no searchValue is specified
      const displayPosts =
        initialDisplayPosts.length > 0 && !searchValue ? initialDisplayPosts : filteredBlogPosts
      return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(jsx_runtime_.Fragment, {
        children: [
          /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
            className: 'divide-y divide-gray-200 dark:divide-gray-700',
            children: [
              /*#__PURE__*/ jsx_runtime_.jsx('div', {
                className: 'space-y-2 pt-6 pb-8 md:space-y-5',
                children:
                  enableSearch &&
                  /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
                    className: ' relative max-w-4xl ',
                    children: [
                      /*#__PURE__*/ jsx_runtime_.jsx('input', {
                        'aria-label': 'Search articles',
                        type: 'text',
                        onChange: (e) => setSearchValue(e.target.value),
                        placeholder: 'Search articles',
                        className:
                          'block w-full rounded-md border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-primary-500 focus:ring-primary-500 dark:border-gray-900 dark:bg-gray-800 dark:text-gray-100',
                      }),
                      /*#__PURE__*/ jsx_runtime_.jsx('svg', {
                        className:
                          'absolute right-3 top-3 h-5 w-5 text-gray-400 dark:text-gray-300',
                        xmlns: 'http://www.w3.org/2000/svg',
                        fill: 'none',
                        viewBox: '0 0 24 24',
                        stroke: 'currentColor',
                        children: /*#__PURE__*/ jsx_runtime_.jsx('path', {
                          strokeLinecap: 'round',
                          strokeLinejoin: 'round',
                          strokeWidth: 2,
                          d: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
                        }),
                      }),
                    ],
                  }),
              }),
              /*#__PURE__*/ (0, jsx_runtime_.jsxs)('ul', {
                children: [
                  !filteredBlogPosts.length && 'No posts found.',
                  displayPosts.map((frontMatter) => {
                    // todo modify this
                    const { slug, date, title, summary, tags } = frontMatter
                    const season = Math.floor((new Date(date).getMonth() / 12) * 4) % 4
                    return /*#__PURE__*/ jsx_runtime_.jsx(
                      'li',
                      {
                        className: 'py-4 font-rs',
                        children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('article', {
                          className:
                            'space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0',
                          children: [
                            /*#__PURE__*/ (0, jsx_runtime_.jsxs)('dl', {
                              children: [
                                /*#__PURE__*/ jsx_runtime_.jsx('dt', {
                                  className: 'sr-only',
                                  children: 'Published on',
                                }),
                                /*#__PURE__*/ (0, jsx_runtime_.jsxs)('dd', {
                                  className:
                                    'text -base font-medium leading-6 text-gray-500 dark:text-gray-400',
                                  children: [
                                    season == 0 &&
                                      /*#__PURE__*/ jsx_runtime_.jsx(
                                        react_fontawesome_.FontAwesomeIcon,
                                        {
                                          icon: 'fan',
                                          className: 'text-pink-300',
                                        }
                                      ),
                                    season == 1 &&
                                      /*#__PURE__*/ jsx_runtime_.jsx(
                                        react_fontawesome_.FontAwesomeIcon,
                                        {
                                          icon: 'sun',
                                          className: 'text-amber-300',
                                        }
                                      ),
                                    season == 2 &&
                                      /*#__PURE__*/ jsx_runtime_.jsx(
                                        react_fontawesome_.FontAwesomeIcon,
                                        {
                                          icon: 'leaf',
                                          className: 'text-green-300',
                                        }
                                      ),
                                    season == 3 &&
                                      /*#__PURE__*/ jsx_runtime_.jsx(
                                        react_fontawesome_.FontAwesomeIcon,
                                        {
                                          icon: 'snowflake',
                                          className: 'text-stone-300',
                                        }
                                      ),
                                    /*#__PURE__*/ jsx_runtime_.jsx('time', {
                                      dateTime: date,
                                      children: ' ' + (0, formatDate /* default */.Z)(date),
                                    }),
                                  ],
                                }),
                              ],
                            }),
                            /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
                              className: 'space-y-3 font-rs xl:col-span-3',
                              children: [
                                /*#__PURE__*/ jsx_runtime_.jsx('div', {
                                  children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
                                    children: [
                                      /*#__PURE__*/ (0, jsx_runtime_.jsxs)('h3', {
                                        className: 'text-2xl font-bold leading-8 tracking-tight ',
                                        children: [
                                          /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                                            href: `/blog/${slug}`,
                                            className:
                                              'text-gray-900 hover:text-primary-600 dark:text-gray-100 dark:hover:text-primary-400',
                                            children: title ? title : slug,
                                          }),
                                          frontMatter.status &&
                                            /*#__PURE__*/ jsx_runtime_.jsx('span', {
                                              className:
                                                'align-top text-sm font-normal text-RSpink',
                                              children: ' [' + frontMatter.status + ']',
                                            }),
                                        ],
                                      }),
                                      frontMatter.subtitle &&
                                        /*#__PURE__*/ jsx_runtime_.jsx('div', {
                                          className: 'text-xl font-normal text-gray-500',
                                          children: frontMatter.subtitle,
                                        }),
                                      /*#__PURE__*/ jsx_runtime_.jsx('div', {
                                        className: 'flex flex-wrap ',
                                        children: tags.map((tag) =>
                                          /*#__PURE__*/ jsx_runtime_.jsx(
                                            Tag /* default */.Z,
                                            {
                                              text: tag,
                                            },
                                            tag
                                          )
                                        ),
                                      }),
                                    ],
                                  }),
                                }),
                                /*#__PURE__*/ jsx_runtime_.jsx('div', {
                                  className: 'prose max-w-none text-gray-500 dark:text-gray-400',
                                  children: summary,
                                }),
                              ],
                            }),
                          ],
                        }),
                      },
                      slug
                    )
                  }),
                ],
              }),
            ],
          }),
          pagination &&
            pagination.totalPages > 1 &&
            !searchValue &&
            /*#__PURE__*/ jsx_runtime_.jsx(Pagination, {
              currentPage: pagination.currentPage,
              totalPages: pagination.totalPages,
            }),
        ],
      })
    }

    /***/
  },

  /***/ 6232: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
      /* harmony export */
    })
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(1576)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_0__)

    const formatDate = (date) => {
      const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      }
      const now = new Date(date).toLocaleDateString(
        _data_siteMetadata__WEBPACK_IMPORTED_MODULE_0___default().locale,
        options
      )
      return now
    }
    /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = formatDate

    /***/
  },
}
