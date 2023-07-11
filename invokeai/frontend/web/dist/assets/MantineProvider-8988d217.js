import{W as d,ga as _,q as h,gj as X}from"./index-078526aa.js";const q={dark:["#C1C2C5","#A6A7AB","#909296","#5c5f66","#373A40","#2C2E33","#25262b","#1A1B1E","#141517","#101113"],gray:["#f8f9fa","#f1f3f5","#e9ecef","#dee2e6","#ced4da","#adb5bd","#868e96","#495057","#343a40","#212529"],red:["#fff5f5","#ffe3e3","#ffc9c9","#ffa8a8","#ff8787","#ff6b6b","#fa5252","#f03e3e","#e03131","#c92a2a"],pink:["#fff0f6","#ffdeeb","#fcc2d7","#faa2c1","#f783ac","#f06595","#e64980","#d6336c","#c2255c","#a61e4d"],grape:["#f8f0fc","#f3d9fa","#eebefa","#e599f7","#da77f2","#cc5de8","#be4bdb","#ae3ec9","#9c36b5","#862e9c"],violet:["#f3f0ff","#e5dbff","#d0bfff","#b197fc","#9775fa","#845ef7","#7950f2","#7048e8","#6741d9","#5f3dc4"],indigo:["#edf2ff","#dbe4ff","#bac8ff","#91a7ff","#748ffc","#5c7cfa","#4c6ef5","#4263eb","#3b5bdb","#364fc7"],blue:["#e7f5ff","#d0ebff","#a5d8ff","#74c0fc","#4dabf7","#339af0","#228be6","#1c7ed6","#1971c2","#1864ab"],cyan:["#e3fafc","#c5f6fa","#99e9f2","#66d9e8","#3bc9db","#22b8cf","#15aabf","#1098ad","#0c8599","#0b7285"],teal:["#e6fcf5","#c3fae8","#96f2d7","#63e6be","#38d9a9","#20c997","#12b886","#0ca678","#099268","#087f5b"],green:["#ebfbee","#d3f9d8","#b2f2bb","#8ce99a","#69db7c","#51cf66","#40c057","#37b24d","#2f9e44","#2b8a3e"],lime:["#f4fce3","#e9fac8","#d8f5a2","#c0eb75","#a9e34b","#94d82d","#82c91e","#74b816","#66a80f","#5c940d"],yellow:["#fff9db","#fff3bf","#ffec99","#ffe066","#ffd43b","#fcc419","#fab005","#f59f00","#f08c00","#e67700"],orange:["#fff4e6","#ffe8cc","#ffd8a8","#ffc078","#ffa94d","#ff922b","#fd7e14","#f76707","#e8590c","#d9480f"]};function Y(r){return()=>({fontFamily:r.fontFamily||"sans-serif"})}var J=Object.defineProperty,x=Object.getOwnPropertySymbols,K=Object.prototype.hasOwnProperty,Q=Object.prototype.propertyIsEnumerable,z=(r,e,o)=>e in r?J(r,e,{enumerable:!0,configurable:!0,writable:!0,value:o}):r[e]=o,j=(r,e)=>{for(var o in e||(e={}))K.call(e,o)&&z(r,o,e[o]);if(x)for(var o of x(e))Q.call(e,o)&&z(r,o,e[o]);return r};function Z(r){return e=>({WebkitTapHighlightColor:"transparent",[e||"&:focus"]:j({},r.focusRing==="always"||r.focusRing==="auto"?r.focusRingStyles.styles(r):r.focusRingStyles.resetStyles(r)),[e?e.replace(":focus",":focus:not(:focus-visible)"):"&:focus:not(:focus-visible)"]:j({},r.focusRing==="auto"||r.focusRing==="never"?r.focusRingStyles.resetStyles(r):null)})}function y(r){return e=>typeof r.primaryShade=="number"?r.primaryShade:r.primaryShade[e||r.colorScheme]}function w(r){const e=y(r);return(o,n,a=!0,t=!0)=>{if(typeof o=="string"&&o.includes(".")){const[s,l]=o.split("."),g=parseInt(l,10);if(s in r.colors&&g>=0&&g<10)return r.colors[s][typeof n=="number"&&!t?n:g]}const i=typeof n=="number"?n:e();return o in r.colors?r.colors[o][i]:a?r.colors[r.primaryColor][i]:o}}function T(r){let e="";for(let o=1;o<r.length-1;o+=1)e+=`${r[o]} ${o/(r.length-1)*100}%, `;return`${r[0]} 0%, ${e}${r[r.length-1]} 100%`}function B(r,...e){return`linear-gradient(${r}deg, ${T(e)})`}function rr(...r){return`radial-gradient(circle, ${T(r)})`}function G(r){const e=w(r),o=y(r);return n=>{const a={from:(n==null?void 0:n.from)||r.defaultGradient.from,to:(n==null?void 0:n.to)||r.defaultGradient.to,deg:(n==null?void 0:n.deg)||r.defaultGradient.deg};return`linear-gradient(${a.deg}deg, ${e(a.from,o(),!1)} 0%, ${e(a.to,o(),!1)} 100%)`}}function D(r){return e=>{if(typeof e=="number")return`${e/16}${r}`;if(typeof e=="string"){const o=e.replace("px","");if(!Number.isNaN(Number(o)))return`${Number(o)/16}${r}`}return e}}const u=D("rem"),k=D("em");function V({size:r,sizes:e,units:o}){return r in e?e[r]:typeof r=="number"?o==="em"?k(r):u(r):r||e.md}function S(r){return typeof r=="number"?r:typeof r=="string"&&r.includes("rem")?Number(r.replace("rem",""))*16:typeof r=="string"&&r.includes("em")?Number(r.replace("em",""))*16:Number(r)}function er(r){return e=>`@media (min-width: ${k(S(V({size:e,sizes:r.breakpoints})))})`}function or(r){return e=>`@media (max-width: ${k(S(V({size:e,sizes:r.breakpoints}))-1)})`}function nr(r){return/^#?([0-9A-F]{3}){1,2}$/i.test(r)}function tr(r){let e=r.replace("#","");if(e.length===3){const i=e.split("");e=[i[0],i[0],i[1],i[1],i[2],i[2]].join("")}const o=parseInt(e,16),n=o>>16&255,a=o>>8&255,t=o&255;return{r:n,g:a,b:t,a:1}}function ar(r){const[e,o,n,a]=r.replace(/[^0-9,.]/g,"").split(",").map(Number);return{r:e,g:o,b:n,a:a||1}}function C(r){return nr(r)?tr(r):r.startsWith("rgb")?ar(r):{r:0,g:0,b:0,a:1}}function p(r,e){if(typeof r!="string"||e>1||e<0)return"rgba(0, 0, 0, 1)";if(r.startsWith("var(--"))return r;const{r:o,g:n,b:a}=C(r);return`rgba(${o}, ${n}, ${a}, ${e})`}function ir(r=0){return{position:"absolute",top:u(r),right:u(r),left:u(r),bottom:u(r)}}function sr(r,e){if(typeof r=="string"&&r.startsWith("var(--"))return r;const{r:o,g:n,b:a,a:t}=C(r),i=1-e,s=l=>Math.round(l*i);return`rgba(${s(o)}, ${s(n)}, ${s(a)}, ${t})`}function lr(r,e){if(typeof r=="string"&&r.startsWith("var(--"))return r;const{r:o,g:n,b:a,a:t}=C(r),i=s=>Math.round(s+(255-s)*e);return`rgba(${i(o)}, ${i(n)}, ${i(a)}, ${t})`}function fr(r){return e=>{if(typeof e=="number")return u(e);const o=typeof r.defaultRadius=="number"?r.defaultRadius:r.radius[r.defaultRadius]||r.defaultRadius;return r.radius[e]||e||o}}function cr(r,e){if(typeof r=="string"&&r.includes(".")){const[o,n]=r.split("."),a=parseInt(n,10);if(o in e.colors&&a>=0&&a<10)return{isSplittedColor:!0,key:o,shade:a}}return{isSplittedColor:!1}}function dr(r){const e=w(r),o=y(r),n=G(r);return({variant:a,color:t,gradient:i,primaryFallback:s})=>{const l=cr(t,r);switch(a){case"light":return{border:"transparent",background:p(e(t,r.colorScheme==="dark"?8:0,s,!1),r.colorScheme==="dark"?.2:1),color:t==="dark"?r.colorScheme==="dark"?r.colors.dark[0]:r.colors.dark[9]:e(t,r.colorScheme==="dark"?2:o("light")),hover:p(e(t,r.colorScheme==="dark"?7:1,s,!1),r.colorScheme==="dark"?.25:.65)};case"subtle":return{border:"transparent",background:"transparent",color:t==="dark"?r.colorScheme==="dark"?r.colors.dark[0]:r.colors.dark[9]:e(t,r.colorScheme==="dark"?2:o("light")),hover:p(e(t,r.colorScheme==="dark"?8:0,s,!1),r.colorScheme==="dark"?.2:1)};case"outline":return{border:e(t,r.colorScheme==="dark"?5:o("light")),background:"transparent",color:e(t,r.colorScheme==="dark"?5:o("light")),hover:r.colorScheme==="dark"?p(e(t,5,s,!1),.05):p(e(t,0,s,!1),.35)};case"default":return{border:r.colorScheme==="dark"?r.colors.dark[4]:r.colors.gray[4],background:r.colorScheme==="dark"?r.colors.dark[6]:r.white,color:r.colorScheme==="dark"?r.white:r.black,hover:r.colorScheme==="dark"?r.colors.dark[5]:r.colors.gray[0]};case"white":return{border:"transparent",background:r.white,color:e(t,o()),hover:null};case"transparent":return{border:"transparent",color:t==="dark"?r.colorScheme==="dark"?r.colors.dark[0]:r.colors.dark[9]:e(t,r.colorScheme==="dark"?2:o("light")),background:"transparent",hover:null};case"gradient":return{background:n(i),color:r.white,border:"transparent",hover:null};default:{const g=o(),$=l.isSplittedColor?l.shade:g,O=l.isSplittedColor?l.key:t;return{border:"transparent",background:e(O,$,s),color:r.white,hover:e(O,$===9?8:$+1)}}}}}function ur(r){return e=>{const o=y(r)(e);return r.colors[r.primaryColor][o]}}function pr(r){return{"@media (hover: hover)":{"&:hover":r},"@media (hover: none)":{"&:active":r}}}function gr(r){return()=>({userSelect:"none",color:r.colorScheme==="dark"?r.colors.dark[3]:r.colors.gray[5]})}function br(r){return()=>r.colorScheme==="dark"?r.colors.dark[2]:r.colors.gray[6]}const f={fontStyles:Y,themeColor:w,focusStyles:Z,linearGradient:B,radialGradient:rr,smallerThan:or,largerThan:er,rgba:p,cover:ir,darken:sr,lighten:lr,radius:fr,variant:dr,primaryShade:y,hover:pr,gradient:G,primaryColor:ur,placeholderStyles:gr,dimmed:br};var mr=Object.defineProperty,yr=Object.defineProperties,Sr=Object.getOwnPropertyDescriptors,R=Object.getOwnPropertySymbols,vr=Object.prototype.hasOwnProperty,_r=Object.prototype.propertyIsEnumerable,F=(r,e,o)=>e in r?mr(r,e,{enumerable:!0,configurable:!0,writable:!0,value:o}):r[e]=o,hr=(r,e)=>{for(var o in e||(e={}))vr.call(e,o)&&F(r,o,e[o]);if(R)for(var o of R(e))_r.call(e,o)&&F(r,o,e[o]);return r},kr=(r,e)=>yr(r,Sr(e));function U(r){return kr(hr({},r),{fn:{fontStyles:f.fontStyles(r),themeColor:f.themeColor(r),focusStyles:f.focusStyles(r),largerThan:f.largerThan(r),smallerThan:f.smallerThan(r),radialGradient:f.radialGradient,linearGradient:f.linearGradient,gradient:f.gradient(r),rgba:f.rgba,cover:f.cover,lighten:f.lighten,darken:f.darken,primaryShade:f.primaryShade(r),radius:f.radius(r),variant:f.variant(r),hover:f.hover,primaryColor:f.primaryColor(r),placeholderStyles:f.placeholderStyles(r),dimmed:f.dimmed(r)}})}const $r={dir:"ltr",primaryShade:{light:6,dark:8},focusRing:"auto",loader:"oval",colorScheme:"light",white:"#fff",black:"#000",defaultRadius:"sm",transitionTimingFunction:"ease",colors:q,lineHeight:1.55,fontFamily:"-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",fontFamilyMonospace:"ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace",primaryColor:"blue",respectReducedMotion:!0,cursorType:"default",defaultGradient:{from:"indigo",to:"cyan",deg:45},shadows:{xs:"0 0.0625rem 0.1875rem rgba(0, 0, 0, 0.05), 0 0.0625rem 0.125rem rgba(0, 0, 0, 0.1)",sm:"0 0.0625rem 0.1875rem rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.05) 0 0.625rem 0.9375rem -0.3125rem, rgba(0, 0, 0, 0.04) 0 0.4375rem 0.4375rem -0.3125rem",md:"0 0.0625rem 0.1875rem rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.05) 0 1.25rem 1.5625rem -0.3125rem, rgba(0, 0, 0, 0.04) 0 0.625rem 0.625rem -0.3125rem",lg:"0 0.0625rem 0.1875rem rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.05) 0 1.75rem 1.4375rem -0.4375rem, rgba(0, 0, 0, 0.04) 0 0.75rem 0.75rem -0.4375rem",xl:"0 0.0625rem 0.1875rem rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.05) 0 2.25rem 1.75rem -0.4375rem, rgba(0, 0, 0, 0.04) 0 1.0625rem 1.0625rem -0.4375rem"},fontSizes:{xs:"0.75rem",sm:"0.875rem",md:"1rem",lg:"1.125rem",xl:"1.25rem"},radius:{xs:"0.125rem",sm:"0.25rem",md:"0.5rem",lg:"1rem",xl:"2rem"},spacing:{xs:"0.625rem",sm:"0.75rem",md:"1rem",lg:"1.25rem",xl:"1.5rem"},breakpoints:{xs:"36em",sm:"48em",md:"62em",lg:"75em",xl:"88em"},headings:{fontFamily:"-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",fontWeight:700,sizes:{h1:{fontSize:"2.125rem",lineHeight:1.3,fontWeight:void 0},h2:{fontSize:"1.625rem",lineHeight:1.35,fontWeight:void 0},h3:{fontSize:"1.375rem",lineHeight:1.4,fontWeight:void 0},h4:{fontSize:"1.125rem",lineHeight:1.45,fontWeight:void 0},h5:{fontSize:"1rem",lineHeight:1.5,fontWeight:void 0},h6:{fontSize:"0.875rem",lineHeight:1.5,fontWeight:void 0}}},other:{},components:{},activeStyles:{transform:"translateY(0.0625rem)"},datesLocale:"en",globalStyles:void 0,focusRingStyles:{styles:r=>({outlineOffset:"0.125rem",outline:`0.125rem solid ${r.colors[r.primaryColor][r.colorScheme==="dark"?7:5]}`}),resetStyles:()=>({outline:"none"}),inputStyles:r=>({outline:"none",borderColor:r.colors[r.primaryColor][typeof r.primaryShade=="object"?r.primaryShade[r.colorScheme]:r.primaryShade]})}},E=U($r);var Pr=Object.defineProperty,wr=Object.defineProperties,Cr=Object.getOwnPropertyDescriptors,H=Object.getOwnPropertySymbols,Er=Object.prototype.hasOwnProperty,Or=Object.prototype.propertyIsEnumerable,M=(r,e,o)=>e in r?Pr(r,e,{enumerable:!0,configurable:!0,writable:!0,value:o}):r[e]=o,xr=(r,e)=>{for(var o in e||(e={}))Er.call(e,o)&&M(r,o,e[o]);if(H)for(var o of H(e))Or.call(e,o)&&M(r,o,e[o]);return r},zr=(r,e)=>wr(r,Cr(e));function jr({theme:r}){return d.createElement(_,{styles:{"*, *::before, *::after":{boxSizing:"border-box"},html:{colorScheme:r.colorScheme==="dark"?"dark":"light"},body:zr(xr({},r.fn.fontStyles()),{backgroundColor:r.colorScheme==="dark"?r.colors.dark[7]:r.white,color:r.colorScheme==="dark"?r.colors.dark[0]:r.black,lineHeight:r.lineHeight,fontSize:r.fontSizes.md,WebkitFontSmoothing:"antialiased",MozOsxFontSmoothing:"grayscale"})}})}function b(r,e,o,n=u){Object.keys(e).forEach(a=>{r[`--mantine-${o}-${a}`]=n(e[a])})}function Rr({theme:r}){const e={"--mantine-color-white":r.white,"--mantine-color-black":r.black,"--mantine-transition-timing-function":r.transitionTimingFunction,"--mantine-line-height":`${r.lineHeight}`,"--mantine-font-family":r.fontFamily,"--mantine-font-family-monospace":r.fontFamilyMonospace,"--mantine-font-family-headings":r.headings.fontFamily,"--mantine-heading-font-weight":`${r.headings.fontWeight}`};b(e,r.shadows,"shadow"),b(e,r.fontSizes,"font-size"),b(e,r.radius,"radius"),b(e,r.spacing,"spacing"),b(e,r.breakpoints,"breakpoints",k),Object.keys(r.colors).forEach(n=>{r.colors[n].forEach((a,t)=>{e[`--mantine-color-${n}-${t}`]=a})});const o=r.headings.sizes;return Object.keys(o).forEach(n=>{e[`--mantine-${n}-font-size`]=o[n].fontSize,e[`--mantine-${n}-line-height`]=`${o[n].lineHeight}`}),d.createElement(_,{styles:{":root":e}})}var Fr=Object.defineProperty,Hr=Object.defineProperties,Mr=Object.getOwnPropertyDescriptors,I=Object.getOwnPropertySymbols,Ir=Object.prototype.hasOwnProperty,Wr=Object.prototype.propertyIsEnumerable,W=(r,e,o)=>e in r?Fr(r,e,{enumerable:!0,configurable:!0,writable:!0,value:o}):r[e]=o,c=(r,e)=>{for(var o in e||(e={}))Ir.call(e,o)&&W(r,o,e[o]);if(I)for(var o of I(e))Wr.call(e,o)&&W(r,o,e[o]);return r},P=(r,e)=>Hr(r,Mr(e));function Ar(r,e){var o;if(!e)return r;const n=Object.keys(r).reduce((a,t)=>{if(t==="headings"&&e.headings){const i=e.headings.sizes?Object.keys(r.headings.sizes).reduce((s,l)=>(s[l]=c(c({},r.headings.sizes[l]),e.headings.sizes[l]),s),{}):r.headings.sizes;return P(c({},a),{headings:P(c(c({},r.headings),e.headings),{sizes:i})})}if(t==="breakpoints"&&e.breakpoints){const i=c(c({},r.breakpoints),e.breakpoints);return P(c({},a),{breakpoints:Object.fromEntries(Object.entries(i).sort((s,l)=>S(s[1])-S(l[1])))})}return a[t]=typeof e[t]=="object"?c(c({},r[t]),e[t]):typeof e[t]=="number"||typeof e[t]=="boolean"||typeof e[t]=="function"?e[t]:e[t]||r[t],a},{});if(e!=null&&e.fontFamily&&!((o=e==null?void 0:e.headings)!=null&&o.fontFamily)&&(n.headings.fontFamily=e.fontFamily),!(n.primaryColor in n.colors))throw new Error("MantineProvider: Invalid theme.primaryColor, it accepts only key of theme.colors, learn more – https://mantine.dev/theming/colors/#primary-color");return n}function Nr(r,e){return U(Ar(r,e))}function Tr(r){return Object.keys(r).reduce((e,o)=>(r[o]!==void 0&&(e[o]=r[o]),e),{})}const Gr={html:{fontFamily:"sans-serif",lineHeight:"1.15",textSizeAdjust:"100%"},body:{margin:0},"article, aside, footer, header, nav, section, figcaption, figure, main":{display:"block"},h1:{fontSize:"2em"},hr:{boxSizing:"content-box",height:0,overflow:"visible"},pre:{fontFamily:"monospace, monospace",fontSize:"1em"},a:{background:"transparent",textDecorationSkip:"objects"},"a:active, a:hover":{outlineWidth:0},"abbr[title]":{borderBottom:"none",textDecoration:"underline"},"b, strong":{fontWeight:"bolder"},"code, kbp, samp":{fontFamily:"monospace, monospace",fontSize:"1em"},dfn:{fontStyle:"italic"},mark:{backgroundColor:"#ff0",color:"#000"},small:{fontSize:"80%"},"sub, sup":{fontSize:"75%",lineHeight:0,position:"relative",verticalAlign:"baseline"},sup:{top:"-0.5em"},sub:{bottom:"-0.25em"},"audio, video":{display:"inline-block"},"audio:not([controls])":{display:"none",height:0},img:{borderStyle:"none",verticalAlign:"middle"},"svg:not(:root)":{overflow:"hidden"},"button, input, optgroup, select, textarea":{fontFamily:"sans-serif",fontSize:"100%",lineHeight:"1.15",margin:0},"button, input":{overflow:"visible"},"button, select":{textTransform:"none"},"button, [type=reset], [type=submit]":{WebkitAppearance:"button"},"button::-moz-focus-inner, [type=button]::-moz-focus-inner, [type=reset]::-moz-focus-inner, [type=submit]::-moz-focus-inner":{borderStyle:"none",padding:0},"button:-moz-focusring, [type=button]:-moz-focusring, [type=reset]:-moz-focusring, [type=submit]:-moz-focusring":{outline:`${u(1)} dotted ButtonText`},legend:{boxSizing:"border-box",color:"inherit",display:"table",maxWidth:"100%",padding:0,whiteSpace:"normal"},progress:{display:"inline-block",verticalAlign:"baseline"},textarea:{overflow:"auto"},"[type=checkbox], [type=radio]":{boxSizing:"border-box",padding:0},"[type=number]::-webkit-inner-spin-button, [type=number]::-webkit-outer-spin-button":{height:"auto"},"[type=search]":{appearance:"none"},"[type=search]::-webkit-search-cancel-button, [type=search]::-webkit-search-decoration":{appearance:"none"},"::-webkit-file-upload-button":{appearance:"button",font:"inherit"},"details, menu":{display:"block"},summary:{display:"list-item"},canvas:{display:"inline-block"},template:{display:"none"}};function Dr(){return d.createElement(_,{styles:Gr})}var Vr=Object.defineProperty,A=Object.getOwnPropertySymbols,Ur=Object.prototype.hasOwnProperty,Lr=Object.prototype.propertyIsEnumerable,N=(r,e,o)=>e in r?Vr(r,e,{enumerable:!0,configurable:!0,writable:!0,value:o}):r[e]=o,m=(r,e)=>{for(var o in e||(e={}))Ur.call(e,o)&&N(r,o,e[o]);if(A)for(var o of A(e))Lr.call(e,o)&&N(r,o,e[o]);return r};const v=h.createContext({theme:E});function L(){var r;return((r=h.useContext(v))==null?void 0:r.theme)||E}function Yr(r){const e=L(),o=n=>{var a,t,i,s;return{styles:((a=e.components[n])==null?void 0:a.styles)||{},classNames:((t=e.components[n])==null?void 0:t.classNames)||{},variants:(i=e.components[n])==null?void 0:i.variants,sizes:(s=e.components[n])==null?void 0:s.sizes}};return Array.isArray(r)?r.map(o):[o(r)]}function Jr(){var r;return(r=h.useContext(v))==null?void 0:r.emotionCache}function Kr(r,e,o){var n;const a=L(),t=(n=a.components[r])==null?void 0:n.defaultProps,i=typeof t=="function"?t(a):t;return m(m(m({},e),i),Tr(o))}function Xr({theme:r,emotionCache:e,withNormalizeCSS:o=!1,withGlobalStyles:n=!1,withCSSVariables:a=!1,inherit:t=!1,children:i}){const s=h.useContext(v),l=Nr(E,t?m(m({},s.theme),r):r);return d.createElement(X,{theme:l},d.createElement(v.Provider,{value:{theme:l,emotionCache:e}},o&&d.createElement(Dr,null),n&&d.createElement(jr,{theme:l}),a&&d.createElement(Rr,{theme:l}),typeof l.globalStyles=="function"&&d.createElement(_,{styles:l.globalStyles(l)}),i))}Xr.displayName="@mantine/core/MantineProvider";export{Xr as M,L as a,Yr as b,V as c,Kr as d,Tr as f,S as g,u as r,Jr as u};
