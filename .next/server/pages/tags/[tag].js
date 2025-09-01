'use strict'
;(() => {
  var exports = {}
  exports.id = 176
  exports.ids = [176]
  exports.modules = {
    /***/ 961: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
      /* harmony export */ __webpack_require__.d(__webpack_exports__, {
        /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
        /* harmony export */
      })
      /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
        __webpack_require__(997)
      /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
        /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
      /* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(1664)
      /* harmony import */ var _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_2__ =
        __webpack_require__(4871)
      /* harmony import */ var _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_3__ =
        __webpack_require__(7197)
      /* harmony import */ var _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_3___default =
        /*#__PURE__*/ __webpack_require__.n(
          _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_3__
        )
      /* harmony import */ var next_image__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(5675)

      const SelectedTag = ({ text }) => {
        if (
          (0, _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z)(text) ===
          'notion'
        ) {
          return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
            next_link__WEBPACK_IMPORTED_MODULE_1__['default'],
            {
              href: `/tags/${(0,
              _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z)(text)}`,
              children: /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                'a',
                {
                  className:
                    'mt-1 mr-3 rounded border-2 border-solid border-primary-500 bg-gray-300 px-2 text-sm font-medium uppercase text-primary-500 transition duration-500 ease-out dark:border-primary-400 dark:bg-gray-500 dark:text-primary-400',
                  children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      next_image__WEBPACK_IMPORTED_MODULE_4__['default'],
                      {
                        className: 'brightness-0 filter dark:brightness-200 dark:filter',
                        src: '/static/images/notion.svg',
                        width: 14,
                        height: 14,
                        alt: 'Notion Blog',
                      }
                    ),
                    ' ' + text.split(' ').join('-'),
                  ],
                }
              ),
            }
          )
        } else if (
          (0, _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z)(text) === 'mdx'
        ) {
          return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
            next_link__WEBPACK_IMPORTED_MODULE_1__['default'],
            {
              href: `/tags/${(0,
              _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z)(text)}`,
              children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('a', {
                className:
                  'mt-1 mr-3 rounded border-2 border-solid border-primary-500 bg-white p-0 text-sm font-medium uppercase text-primary-500 transition duration-500 ease-out dark:border-primary-400 dark:text-primary-400',
                children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                  next_image__WEBPACK_IMPORTED_MODULE_4__['default'],
                  {
                    src: '/static/images/mdx.png',
                    width: 34,
                    height: 14,
                    alt: 'mdx',
                  }
                ),
              }),
            }
          )
        } else {
          return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
            next_link__WEBPACK_IMPORTED_MODULE_1__['default'],
            {
              href: `/tags/${(0,
              _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z)(text)}`,
              children: /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                'a',
                {
                  className:
                    'mt-1 mr-3 rounded border-2 border-solid border-primary-500 bg-gray-300 px-2 text-sm font-medium uppercase text-primary-500 transition duration-500 ease-out dark:border-primary-400 dark:bg-gray-500 dark:text-primary-400',
                  children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_3__.FontAwesomeIcon,
                      {
                        icon: 'tags',
                        className: 'text-black dark:text-gray-300 ',
                      }
                    ),
                    ' ' + text.split(' ').join('-'),
                  ],
                }
              ),
            }
          )
        }
      }
      /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = SelectedTag

      /***/
    },

    /***/ 6888: /***/ (module, __webpack_exports__, __webpack_require__) => {
      __webpack_require__.a(
        module,
        async (__webpack_handle_async_dependencies__, __webpack_async_result__) => {
          try {
            __webpack_require__.r(__webpack_exports__)
            /* harmony export */ __webpack_require__.d(__webpack_exports__, {
              /* harmony export */ getStaticPaths: () => /* binding */ getStaticPaths,
              /* harmony export */ getStaticProps: () => /* binding */ getStaticProps,
              /* harmony export */ default: () => /* binding */ TagPage,
              /* harmony export */
            })
            /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
              __webpack_require__(997)
            /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
              /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
            /* harmony import */ var _components_SEO__WEBPACK_IMPORTED_MODULE_1__ =
              __webpack_require__(9831)
            /* harmony import */ var _components_Link__WEBPACK_IMPORTED_MODULE_2__ =
              __webpack_require__(7233)
            /* harmony import */ var _components_Tag__WEBPACK_IMPORTED_MODULE_3__ =
              __webpack_require__(9019)
            /* harmony import */ var _components_SelectedTag__WEBPACK_IMPORTED_MODULE_4__ =
              __webpack_require__(961)
            /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_5__ =
              __webpack_require__(1576)
            /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_5___default =
              /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_5__)
            /* harmony import */ var _layouts_ListLayout__WEBPACK_IMPORTED_MODULE_6__ =
              __webpack_require__(6055)
            /* harmony import */ var _lib_generate_rss__WEBPACK_IMPORTED_MODULE_7__ =
              __webpack_require__(642)
            /* harmony import */ var _lib_mdx__WEBPACK_IMPORTED_MODULE_8__ =
              __webpack_require__(9882)
            /* harmony import */ var _lib_tags__WEBPACK_IMPORTED_MODULE_9__ =
              __webpack_require__(8234)
            /* harmony import */ var _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_10__ =
              __webpack_require__(4871)
            /* harmony import */ var fs__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(7147)
            /* harmony import */ var fs__WEBPACK_IMPORTED_MODULE_11___default =
              /*#__PURE__*/ __webpack_require__.n(fs__WEBPACK_IMPORTED_MODULE_11__)
            /* harmony import */ var path__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(1017)
            /* harmony import */ var path__WEBPACK_IMPORTED_MODULE_12___default =
              /*#__PURE__*/ __webpack_require__.n(path__WEBPACK_IMPORTED_MODULE_12__)
            var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([
              _lib_mdx__WEBPACK_IMPORTED_MODULE_8__,
              _lib_tags__WEBPACK_IMPORTED_MODULE_9__,
            ])
            ;[_lib_mdx__WEBPACK_IMPORTED_MODULE_8__, _lib_tags__WEBPACK_IMPORTED_MODULE_9__] =
              __webpack_async_dependencies__.then
                ? (await __webpack_async_dependencies__)()
                : __webpack_async_dependencies__

            const root = process.cwd()
            async function getStaticPaths() {
              const tags = await (0, _lib_tags__WEBPACK_IMPORTED_MODULE_9__ /* .getAllTags */.Y)(
                'blog'
              )
              return {
                paths: Object.keys(tags).map((tag) => ({
                  params: {
                    tag,
                  },
                })),
                fallback: false,
              }
            }
            async function getStaticProps({ params }) {
              const allPosts = await (0,
              _lib_mdx__WEBPACK_IMPORTED_MODULE_8__ /* .getAllFilesFrontMatter */.sj)('blog')
              const all_tags = await (0,
              _lib_tags__WEBPACK_IMPORTED_MODULE_9__ /* .getAllTags */.Y)('blog')
              const filteredPosts = allPosts.filter(
                (post) =>
                  post.draft !== true &&
                  post.tags
                    .map((t) =>
                      (0, _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_10__ /* ["default"] */.Z)(t)
                    )
                    .includes(params.tag)
              )
              // rss
              if (filteredPosts.length > 0) {
                const rss = (0, _lib_generate_rss__WEBPACK_IMPORTED_MODULE_7__ /* ["default"] */.Z)(
                  filteredPosts,
                  `tags/${params.tag}/feed.xml`
                )
                const rssPath = path__WEBPACK_IMPORTED_MODULE_12___default().join(
                  root,
                  'public',
                  'tags',
                  params.tag
                )
                fs__WEBPACK_IMPORTED_MODULE_11___default().mkdirSync(rssPath, {
                  recursive: true,
                })
                fs__WEBPACK_IMPORTED_MODULE_11___default().writeFileSync(
                  path__WEBPACK_IMPORTED_MODULE_12___default().join(rssPath, 'feed.xml'),
                  rss
                )
              }
              return {
                props: {
                  posts: filteredPosts,
                  tag: params.tag,
                  all_tags: all_tags,
                },
              }
            }
            function TagPage({ posts, tag, all_tags }) {
              // Capitalize first letter and convert space to dash
              const title = tag[0].toUpperCase() + tag.split(' ').join('-').slice(1)
              // const title = 'Title'
              const sortedTags = Object.keys(all_tags).sort((a, b) => all_tags[b] - all_tags[a])
              // console.log('tag',tag)
              // console.log('tags',sortedTags )
              return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                {
                  children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      _components_SEO__WEBPACK_IMPORTED_MODULE_1__ /* .TagSEO */.$t,
                      {
                        title: `${tag} - ${
                          _data_siteMetadata__WEBPACK_IMPORTED_MODULE_5___default().author
                        }`,
                        description: `${tag} tags - ${
                          _data_siteMetadata__WEBPACK_IMPORTED_MODULE_5___default().author
                        }`,
                      }
                    ),
                    /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                      className:
                        'flex flex-col items-start justify-start divide-y divide-gray-200 dark:divide-gray-700 md:mt-24 md:flex-row md:items-center md:justify-center md:space-x-6 md:divide-y-0',
                      children: [
                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
                          className: 'space-x-2 pt-6 pb-8 md:space-y-5',
                          children:
                            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('h1', {
                              className:
                                'text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:border-r-2 md:px-6 md:text-6xl md:leading-14',
                              children: 'Tags',
                            }),
                        }),
                        /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                          'div',
                          {
                            className: 'flex max-w-lg flex-wrap',
                            children: [
                              Object.keys(all_tags).length === 0 && 'No tags found.',
                              sortedTags.map((t) => {
                                return /*#__PURE__*/ (0,
                                react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                                  'div',
                                  {
                                    className: 'mt-2 mb-2 mr-5',
                                    children: [
                                      t === tag
                                        ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                            _components_SelectedTag__WEBPACK_IMPORTED_MODULE_4__ /* ["default"] */.Z,
                                            {
                                              text: t,
                                            }
                                          )
                                        : /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                            _components_Tag__WEBPACK_IMPORTED_MODULE_3__ /* ["default"] */.Z,
                                            {
                                              text: t,
                                            }
                                          ),
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        _components_Link__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z,
                                        {
                                          href: `/tags/${(0,
                                          _lib_utils_kebabCase__WEBPACK_IMPORTED_MODULE_10__ /* ["default"] */.Z)(
                                            t
                                          )}`,
                                          className:
                                            '-ml-2 text-sm font-semibold uppercase text-gray-600 dark:text-gray-300',
                                          children: ` (${all_tags[t]})`,
                                        }
                                      ),
                                    ],
                                  },
                                  t
                                )
                              }),
                            ],
                          }
                        ),
                      ],
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      _layouts_ListLayout__WEBPACK_IMPORTED_MODULE_6__['default'],
                      {
                        posts: posts,
                        title: title,
                        enableSearch: false,
                      }
                    ),
                  ],
                }
              )
            }

            __webpack_async_result__()
          } catch (e) {
            __webpack_async_result__(e)
          }
        }
      )

      /***/
    },

    /***/ 7197: /***/ (module) => {
      module.exports = require('@fortawesome/react-fontawesome')

      /***/
    },

    /***/ 8904: /***/ (module) => {
      module.exports = require('github-slugger')

      /***/
    },

    /***/ 8076: /***/ (module) => {
      module.exports = require('gray-matter')

      /***/
    },

    /***/ 7219: /***/ (module) => {
      module.exports = require('image-size')

      /***/
    },

    /***/ 9793: /***/ (module) => {
      module.exports = require('js-yaml')

      /***/
    },

    /***/ 8214: /***/ (module) => {
      module.exports = require('mdx-bundler')

      /***/
    },

    /***/ 562: /***/ (module) => {
      module.exports = require('next/dist/server/denormalize-page-path.js')

      /***/
    },

    /***/ 4957: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/head.js')

      /***/
    },

    /***/ 4014: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/i18n/normalize-locale-path.js')

      /***/
    },

    /***/ 744: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/image-config-context.js')

      /***/
    },

    /***/ 5843: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/image-config.js')

      /***/
    },

    /***/ 8524: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/is-plain-object.js')

      /***/
    },

    /***/ 8020: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/mitt.js')

      /***/
    },

    /***/ 4964: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router-context.js')

      /***/
    },

    /***/ 3938: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/format-url.js')

      /***/
    },

    /***/ 9565: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/get-asset-path-from-route.js')

      /***/
    },

    /***/ 1428: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/is-dynamic.js')

      /***/
    },

    /***/ 1292: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/parse-relative-url.js')

      /***/
    },

    /***/ 979: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/querystring.js')

      /***/
    },

    /***/ 6052: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/resolve-rewrites.js')

      /***/
    },

    /***/ 4226: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/route-matcher.js')

      /***/
    },

    /***/ 5052: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/route-regex.js')

      /***/
    },

    /***/ 4241: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/router/utils/routing-items.js')

      /***/
    },

    /***/ 9232: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/utils.js')

      /***/
    },

    /***/ 968: /***/ (module) => {
      module.exports = require('next/head')

      /***/
    },

    /***/ 1853: /***/ (module) => {
      module.exports = require('next/router')

      /***/
    },

    /***/ 6689: /***/ (module) => {
      module.exports = require('react')

      /***/
    },

    /***/ 997: /***/ (module) => {
      module.exports = require('react/jsx-runtime')

      /***/
    },

    /***/ 4956: /***/ (module) => {
      module.exports = require('reading-time')

      /***/
    },

    /***/ 3614: /***/ (module) => {
      module.exports = import('mdast-util-to-string')

      /***/
    },

    /***/ 3861: /***/ (module) => {
      module.exports = import('mdx-mermaid')

      /***/
    },

    /***/ 3467: /***/ (module) => {
      module.exports = import('mdx-mermaid/lib/Mermaid')

      /***/
    },

    /***/ 9847: /***/ (module) => {
      module.exports = import('rehype-autolink-headings')

      /***/
    },

    /***/ 1380: /***/ (module) => {
      module.exports = import('rehype-citation')

      /***/
    },

    /***/ 9521: /***/ (module) => {
      module.exports = import('rehype-katex')

      /***/
    },

    /***/ 6370: /***/ (module) => {
      module.exports = import('rehype-preset-minify')

      /***/
    },

    /***/ 483: /***/ (module) => {
      module.exports = import('rehype-prism-plus')

      /***/
    },

    /***/ 7752: /***/ (module) => {
      module.exports = import('rehype-slug')

      /***/
    },

    /***/ 1083: /***/ (module) => {
      module.exports = import('remark-footnotes')

      /***/
    },

    /***/ 6809: /***/ (module) => {
      module.exports = import('remark-gfm')

      /***/
    },

    /***/ 9832: /***/ (module) => {
      module.exports = import('remark-math')

      /***/
    },

    /***/ 6016: /***/ (module) => {
      module.exports = import('unist-util-visit')

      /***/
    },

    /***/ 7147: /***/ (module) => {
      module.exports = require('fs')

      /***/
    },

    /***/ 1017: /***/ (module) => {
      module.exports = require('path')

      /***/
    },
  }
  // load runtime
  var __webpack_require__ = require('../../webpack-runtime.js')
  __webpack_require__.C(exports)
  var __webpack_exec__ = (moduleId) => __webpack_require__((__webpack_require__.s = moduleId))
  var __webpack_exports__ = __webpack_require__.X(
    0,
    [895, 664, 675, 776, 831, 397, 159, 55, 642, 234],
    () => __webpack_exec__(6888)
  )
  module.exports = __webpack_exports__
})()
