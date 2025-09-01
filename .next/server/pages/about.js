'use strict'
;(() => {
  var exports = {}
  exports.id = 521
  exports.ids = [521]
  exports.modules = {
    /***/ 8953: /***/ (module, __webpack_exports__, __webpack_require__) => {
      __webpack_require__.a(
        module,
        async (__webpack_handle_async_dependencies__, __webpack_async_result__) => {
          try {
            __webpack_require__.r(__webpack_exports__)
            /* harmony export */ __webpack_require__.d(__webpack_exports__, {
              /* harmony export */ getStaticProps: () => /* binding */ getStaticProps,
              /* harmony export */ default: () => /* binding */ About,
              /* harmony export */
            })
            /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
              __webpack_require__(997)
            /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
              /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
            /* harmony import */ var _components_MDXComponents__WEBPACK_IMPORTED_MODULE_1__ =
              __webpack_require__(7751)
            /* harmony import */ var _lib_mdx__WEBPACK_IMPORTED_MODULE_2__ =
              __webpack_require__(9882)
            /* harmony import */ var notion_client__WEBPACK_IMPORTED_MODULE_3__ =
              __webpack_require__(743)
            /* harmony import */ var react_notion_x__WEBPACK_IMPORTED_MODULE_4__ =
              __webpack_require__(5574)
            var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([
              _components_MDXComponents__WEBPACK_IMPORTED_MODULE_1__,
              _lib_mdx__WEBPACK_IMPORTED_MODULE_2__,
              notion_client__WEBPACK_IMPORTED_MODULE_3__,
              react_notion_x__WEBPACK_IMPORTED_MODULE_4__,
            ])
            ;[
              _components_MDXComponents__WEBPACK_IMPORTED_MODULE_1__,
              _lib_mdx__WEBPACK_IMPORTED_MODULE_2__,
              notion_client__WEBPACK_IMPORTED_MODULE_3__,
              react_notion_x__WEBPACK_IMPORTED_MODULE_4__,
            ] = __webpack_async_dependencies__.then
              ? (await __webpack_async_dependencies__)()
              : __webpack_async_dependencies__

            const DEFAULT_LAYOUT = 'AuthorLayout'
            // const DEFAULT_LAYOUT = 'NotionLayout'
            //Next.js SSR
            async function getStaticProps() {
              const aboutDetails = await (0,
              _lib_mdx__WEBPACK_IMPORTED_MODULE_2__ /* .getFileBySlug */.x7)('about', ['default'])
              //notion
              let recordMap = null
              if (aboutDetails.frontMatter.notion) {
                const notion = new notion_client__WEBPACK_IMPORTED_MODULE_3__.NotionAPI()
                recordMap = await notion.getPage(aboutDetails.frontMatter.notion)
              } else {
                recordMap = null
              }
              return {
                props: {
                  aboutDetails,
                  recordMap,
                },
              }
            }
            function About({ aboutDetails, recordMap }) {
              const { mdxSource, frontMatter } = aboutDetails
              return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                _components_MDXComponents__WEBPACK_IMPORTED_MODULE_1__ /* .MDXLayoutRenderer */.J,
                {
                  layout: frontMatter.layout || DEFAULT_LAYOUT,
                  mdxSource: mdxSource,
                  recordMap: recordMap,
                  frontMatter: frontMatter,
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

    /***/ 1618: /***/ (module) => {
      module.exports = require('@matejmazur/react-katex')

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

    /***/ 1228: /***/ (module) => {
      module.exports = require('mdx-bundler/client')

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

    /***/ 5832: /***/ (module) => {
      module.exports = require('next/dist/shared/lib/loadable.js')

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

    /***/ 258: /***/ (module) => {
      module.exports = require('react-fast-compare')

      /***/
    },

    /***/ 2784: /***/ (module) => {
      module.exports = require('react-hotkeys-hook')

      /***/
    },

    /***/ 9358: /***/ (module) => {
      module.exports = require('react-image')

      /***/
    },

    /***/ 9785: /***/ (module) => {
      module.exports = require('react-intersection-observer')

      /***/
    },

    /***/ 9755: /***/ (module) => {
      module.exports = require('react-use')

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

    /***/ 3470: /***/ (module) => {
      module.exports = require('unionize')

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

    /***/ 743: /***/ (module) => {
      module.exports = import('notion-client')

      /***/
    },

    /***/ 8751: /***/ (module) => {
      module.exports = import('notion-utils')

      /***/
    },

    /***/ 5574: /***/ (module) => {
      module.exports = import('react-notion-x')

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
  var __webpack_require__ = require('../webpack-runtime.js')
  __webpack_require__.C(exports)
  var __webpack_exec__ = (moduleId) => __webpack_require__((__webpack_require__.s = moduleId))
  var __webpack_exports__ = __webpack_require__.X(
    0,
    [895, 664, 675, 152, 776, 831, 397, 159, 55, 726, 751],
    () => __webpack_exec__(8953)
  )
  module.exports = __webpack_exports__
})()
