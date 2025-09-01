'use strict'
exports.id = 540
exports.ids = [540]
exports.modules = {
  /***/ 5540: /***/ (module, __webpack_exports__, __webpack_require__) => {
    __webpack_require__.a(
      module,
      async (__webpack_handle_async_dependencies__, __webpack_async_result__) => {
        try {
          __webpack_require__.r(__webpack_exports__)
          /* harmony export */ __webpack_require__.d(__webpack_exports__, {
            /* harmony export */ POSTS_PER_PAGE: () => /* binding */ POSTS_PER_PAGE,
            /* harmony export */ getStaticProps: () => /* binding */ getStaticProps,
            /* harmony export */ default: () => /* binding */ Blog,
            /* harmony export */
          })
          /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
            __webpack_require__(997)
          /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
            /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
          /* harmony import */ var _lib_mdx__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(9882)
          /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_2__ =
            __webpack_require__(1576)
          /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_2___default =
            /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_2__)
          /* harmony import */ var _layouts_ListLayout__WEBPACK_IMPORTED_MODULE_3__ =
            __webpack_require__(6055)
          /* harmony import */ var _components_SEO__WEBPACK_IMPORTED_MODULE_4__ =
            __webpack_require__(9831)
          /* harmony import */ var _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_5__ =
            __webpack_require__(7197)
          /* harmony import */ var _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_5___default =
            /*#__PURE__*/ __webpack_require__.n(
              _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_5__
            )
          /* harmony import */ var _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_6__ =
            __webpack_require__(4563)
          var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([
            _lib_mdx__WEBPACK_IMPORTED_MODULE_1__,
            _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_6__,
          ])
          ;[
            _lib_mdx__WEBPACK_IMPORTED_MODULE_1__,
            _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_6__,
          ] = __webpack_async_dependencies__.then
            ? (await __webpack_async_dependencies__)()
            : __webpack_async_dependencies__

          const POSTS_PER_PAGE = 20
          async function getStaticProps() {
            const posts = await (0,
            _lib_mdx__WEBPACK_IMPORTED_MODULE_1__ /* .getAllFilesFrontMatter */.sj)('blog')
            const initialDisplayPosts = posts.slice(0, POSTS_PER_PAGE)
            const pagination = {
              currentPage: 1,
              totalPages: Math.ceil(posts.length / POSTS_PER_PAGE),
            }
            return {
              props: {
                initialDisplayPosts,
                posts,
                pagination,
              },
            }
          }
          function Blog({ posts, initialDisplayPosts, pagination }) {
            return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
              react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              {
                children: [
                  /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                    _components_SEO__WEBPACK_IMPORTED_MODULE_4__ /* .PageSEO */.TQ,
                    {
                      title: `Blog - ${
                        _data_siteMetadata__WEBPACK_IMPORTED_MODULE_2___default().author
                      }`,
                      description:
                        _data_siteMetadata__WEBPACK_IMPORTED_MODULE_2___default().description,
                    }
                  ),
                  /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                    _layouts_ListLayout__WEBPACK_IMPORTED_MODULE_3__['default'],
                    {
                      posts: posts,
                      initialDisplayPosts: initialDisplayPosts,
                      pagination: pagination,
                      title: 'All Posts',
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
}
