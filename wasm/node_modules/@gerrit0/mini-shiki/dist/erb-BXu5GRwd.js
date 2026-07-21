import html from './html-DJjKhWOc.js';
import ruby from './ruby-B35gCjOI.js';
import './javascript-CjVMUwBu.js';
import './css-CcnpzXBr.js';
import './haml-BD7qFoCJ.js';
import './xml-DNQ2AKLY.js';
import './java-J9ifFwHK.js';
import './sql-CwwRRajN.js';
import './graphql-DqfhDByg.js';
import './typescript-DFjkS2ah.js';
import './jsx-C6hsJK1G.js';
import './tsx-zCkZ74Uo.js';
import './cpp-Do3vLwZR.js';
import './regexp-D6QN1sWt.js';
import './glsl-DNxBKKNq.js';
import './c-DWyntzpU.js';
import './shellscript-DvfPQ4T_.js';
import './lua-CBfHEbQR.js';
import './yaml-DfQjDO74.js';

const lang = Object.freeze(JSON.parse("{\"displayName\":\"ERB\",\"fileTypes\":[\"erb\",\"rhtml\",\"html.erb\"],\"injections\":{\"text.html.erb - (meta.embedded.block.erb | meta.embedded.line.erb | comment)\":{\"patterns\":[{\"begin\":\"(^\\\\s*)(?=<%+#(?![^%]*%>))\",\"beginCaptures\":{\"0\":{\"name\":\"punctuation.whitespace.comment.leading.erb\"}},\"end\":\"(?!\\\\G)(\\\\s*$\\\\n)?\",\"endCaptures\":{\"0\":{\"name\":\"punctuation.whitespace.comment.trailing.erb\"}},\"patterns\":[{\"include\":\"#comment\"}]},{\"begin\":\"(^\\\\s*)(?=<%(?![^%]*%>))\",\"beginCaptures\":{\"0\":{\"name\":\"punctuation.whitespace.embedded.leading.erb\"}},\"end\":\"(?!\\\\G)(\\\\s*$\\\\n)?\",\"endCaptures\":{\"0\":{\"name\":\"punctuation.whitespace.embedded.trailing.erb\"}},\"patterns\":[{\"include\":\"#tags\"}]},{\"include\":\"#comment\"},{\"include\":\"#tags\"}]}},\"name\":\"erb\",\"patterns\":[{\"include\":\"text.html.basic\"}],\"repository\":{\"comment\":{\"patterns\":[{\"begin\":\"<%+#\",\"beginCaptures\":{\"0\":{\"name\":\"punctuation.definition.comment.begin.erb\"}},\"end\":\"%>\",\"endCaptures\":{\"0\":{\"name\":\"punctuation.definition.comment.end.erb\"}},\"name\":\"comment.block.erb\"}]},\"tags\":{\"patterns\":[{\"begin\":\"<%+(?!>)[-=]?(?![^%]*%>)\",\"beginCaptures\":{\"0\":{\"name\":\"punctuation.section.embedded.begin.erb\"}},\"contentName\":\"source.ruby\",\"end\":\"(-?%)>\",\"endCaptures\":{\"0\":{\"name\":\"punctuation.section.embedded.end.erb\"},\"1\":{\"name\":\"source.ruby\"}},\"name\":\"meta.embedded.block.erb\",\"patterns\":[{\"captures\":{\"1\":{\"name\":\"punctuation.definition.comment.erb\"}},\"match\":\"(#).*?(?=-?%>)\",\"name\":\"comment.line.number-sign.erb\"},{\"include\":\"source.ruby\"}]},{\"begin\":\"<%+(?!>)[-=]?\",\"beginCaptures\":{\"0\":{\"name\":\"punctuation.section.embedded.begin.erb\"}},\"contentName\":\"source.ruby\",\"end\":\"(-?%)>\",\"endCaptures\":{\"0\":{\"name\":\"punctuation.section.embedded.end.erb\"},\"1\":{\"name\":\"source.ruby\"}},\"name\":\"meta.embedded.line.erb\",\"patterns\":[{\"captures\":{\"1\":{\"name\":\"punctuation.definition.comment.erb\"}},\"match\":\"(#).*?(?=-?%>)\",\"name\":\"comment.line.number-sign.erb\"},{\"include\":\"source.ruby\"}]}]}},\"scopeName\":\"text.html.erb\",\"embeddedLangs\":[\"html\",\"ruby\"]}"));

var erb = [
...html,
...ruby,
lang
];

export { erb as default };
