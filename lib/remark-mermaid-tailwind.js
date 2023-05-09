import { visit } from 'unist-util-visit'

export default function remarkMermaidTailwind() {
  return (tree) => {
    visit(
      tree,
      (node) => node.type === 'code'
      // (node) => {console.log(node)}
    )
  }
}
