// The Python editor used by every runnable cell: CodeMirror 6 with Python highlighting, line numbers,
// bracket matching, auto-indent and undo. Loaded on demand (see CodeEditor in python.tsx), so pages
// without code never download it. Colours follow the course's code theme (--code-bg, --code-ink).
import { EditorState } from '@codemirror/state'
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection } from '@codemirror/view'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { bracketMatching, indentOnInput, syntaxHighlighting, HighlightStyle, indentUnit } from '@codemirror/language'
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete'
import { python } from '@codemirror/lang-python'
import { tags as t } from '@lezer/highlight'

const courseHighlight = HighlightStyle.define([
  { tag: t.comment, color: '#8689ad', fontStyle: 'italic' },
  { tag: [t.string, t.special(t.string)], color: '#8fe3bd' },
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword, t.bool, t.null, t.self], color: '#f3a6c8' },
  { tag: [t.number], color: '#ffc98a' },
  { tag: [t.function(t.variableName), t.function(t.propertyName), t.definition(t.function(t.variableName))], color: '#a9abff' },
  { tag: [t.className, t.definition(t.className)], color: '#ffd89a' },
])

const courseTheme = EditorView.theme({
  '&': { backgroundColor: 'var(--code-bg)', color: 'var(--code-ink)', fontSize: '13.5px' },
  '.cm-content': { fontFamily: 'var(--mono)', padding: '12px 0', caretColor: '#ffffff' },
  '.cm-scroller': { fontFamily: 'var(--mono)', lineHeight: '1.65' },
  '.cm-gutters': { backgroundColor: 'var(--code-bg)', color: '#5d6080', border: 'none', paddingLeft: '6px' },
  '.cm-activeLine': { backgroundColor: 'rgba(255, 255, 255, 0.04)' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: '#b9bbd6' },
  '&.cm-focused': { outline: 'none' },
  '&.cm-focused .cm-cursor': { borderLeftColor: '#ffffff' },
  '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': { backgroundColor: 'rgba(142, 144, 255, 0.35) !important' },
  '.cm-matchingBracket': { backgroundColor: 'rgba(143, 227, 189, 0.18)', outline: '1px solid rgba(143, 227, 189, 0.45)' },
}, { dark: true })

export interface PyEditorOptions {
  doc: string
  label: string
  minLines: number
  onChange: (v: string) => void
  /** Ctrl/Cmd+Enter runs the cell. */
  onRun?: () => void
}

export function createPythonEditor(parent: HTMLElement, o: PyEditorOptions) {
  let onChange = o.onChange
  let onRun = o.onRun
  const view = new EditorView({
    parent,
    state: EditorState.create({
      doc: o.doc,
      extensions: [
        lineNumbers(),
        highlightActiveLineGutter(),
        highlightActiveLine(),
        drawSelection(),
        history(),
        indentOnInput(),
        indentUnit.of('    '),
        bracketMatching(),
        closeBrackets(),
        python(),
        syntaxHighlighting(courseHighlight),
        courseTheme,
        // Tab indents; Escape then Tab leaves the editor, so keyboard users are never trapped.
        keymap.of([{ key: 'Mod-Enter', run: () => { onRun?.(); return true } }, ...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
        EditorView.contentAttributes.of({ 'aria-label': o.label, 'aria-description': 'Python editor. Ctrl or Cmd plus Enter runs it. Press Escape and then Tab to leave the editor.' }),
        EditorView.theme({ '.cm-content, .cm-gutter': { minHeight: `${o.minLines * 1.65 * 13.5 + 24}px` } }),
        EditorView.updateListener.of((u) => { if (u.docChanged) onChange(u.state.doc.toString()) }),
      ],
    }),
  })
  return {
    view,
    /** Replace the text from outside (Reset, Start over) without echoing it back as an edit. */
    setValue(v: string) {
      if (v === view.state.doc.toString()) return
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: v } })
    },
    setHandlers(change: (v: string) => void, run?: () => void) { onChange = change; onRun = run },
    destroy() { view.destroy() },
  }
}
