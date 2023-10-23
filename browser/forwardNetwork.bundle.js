!function(t,n){if("object"==typeof exports&&"object"==typeof module)module.exports=n();else if("function"==typeof define&&define.amd)define([],n);else{var r=n();for(var e in r)("object"==typeof exports?exports:t)[e]=r[e]}}(self,(()=>(()=>{"use strict";var t={423:(t,n,r)=>{const e=r(673),i=r(276);function o(t){let n=[];for(;Array.isArray(t);)n.push(t.length),t=t[0];return n}function a(t){return o(t).length}function u(t,n,r){if(!n.length)return t;if(2===n.length){let e=n[0]>n[1],i=e?n[1]:n[0],o=e?n[0]:n[1];return a(t)>1&&r>0?t.map((function(t){return e?t.slice(i,o+1).reverse():t.slice(i,o+1)})):(t=t.slice(i,o+1),e&&t.reverse()||t)}}function s(t,n,r){let e=[],i=n();for(let n=0;n<t.length;n++){let o=t[n],a=i[n];e.push(o.map((function(t,n){return r(t,a[n])})))}return e}function c(t,n){return t+n}function l(t,n){return t-n}function f(t,n){return t*n}function h(t,n){return t/n}function y(t,n,r){let e=t[n];t[n]=t[r],t[r]=e}function p(t){let n=[];return t.forEach(((t,r)=>{n.push(t.map((t=>e(t))))})),n}t.exports=function(t){if(!Array.isArray(t))throw new Error("Input should be of type array");return Object.assign((function(){let n=1===arguments.length?[arguments[0]]:Array.apply(null,arguments);return function(t,n){return 0===n.length?t:function(t,n){let r=a(t);for(let e=0;e<r;e++){let r=n[e];if(void 0===r)break;Array.isArray(r)?t=u(t,r,e):Number.isInteger(r)&&(t=a(t)>1&&e>0?t.map((function(t){return[t[r]]})):t[r])}return t}(t,n)}(t,n)}),function(t){return{size:()=>o(t),add:n=>s(t,n,c),sub:n=>s(t,n,l),mul:n=>s(t,n,f),div:n=>s(t,n,h),prod:n=>function(t,n){let r=t,e=n(),i=o(r),a=o(e),u=[];if(i[1]===a[0])for(let t=0;t<i[0];t++){u[t]=[];for(let n=0;n<a[1];n++)for(let o=0;o<i[1];o++)void 0===u[t][n]&&(u[t][n]=0),u[t][n]+=f(r[t][o],e[o][n])}return u}(t,n),trans:()=>function(t){let n=t,r=o(t),e=[];for(let t=0;t<r[0];t++)for(let i=0;i<r[1];i++)Array.isArray(e[i])?e[i].push(n[t][i]):e[i]=[n[t][i]];return e}(t),set:function(){let n=1===arguments.length?[arguments[0]]:Array.apply(null,arguments);return{to:r=>function(t,n,r){let e=function(t){let n=[];for(let r=0;r<t.length;r++)n.push(t[r].slice(0));return n}(t),i=r[0],o=i[0]||0,a=i[1]&&i[1]+1||t.length;if(Array.isArray(i)||1!==r.length){if(1===r.length)for(let t=o;t<a;t++)e[t].fill(n)}else e[i].fill(n);for(let u=1;u<r.length;u++){let s=Array.isArray(r[u])?r[u][0]||0:r[u],c=Array.isArray(r[u])?r[u][1]&&r[u][1]+1||t[0].length:r[u]+1;if(Array.isArray(i))for(let t=o;t<a;t++)e[t].fill(n,s,c);else e[i].fill(n,s,c)}return e}(t,r,n)}},det:()=>function(t){let n=p(t),r=o(t),i=e(1),a=1;for(let t=0;t<r[0]-1;t++)for(let i=t+1;i<r[0];i++){if(0===n[i][t].num)continue;if(0===n[t][t].num){y(n,t,i),a=-a;continue}let o=n[i][t].div(n[t][t]);o=e(Math.abs(o.num),o.den),Math.sign(n[i][t].num)===Math.sign(n[t][t].num)&&(o=e(-o.num,o.den));for(let e=0;e<r[1];e++)n[i][e]=o.mul(n[t][e]).add(n[i][e])}return i=n.reduce(((t,n,r)=>t.mul(n[r])),e(1)),a*i.num/i.den}(t),inv:()=>function(t){let n=p(t),r=o(t),i=p(function(t){let n=function(t,n){let r=2;for(;r>0;){for(var e=[],i=0;i<t;i++)Array.isArray(n)?e.push(Object.assign([],n)):e.push(n);n=e,r-=1}return n}(t,0);return n.forEach(((t,n)=>{t[n]=1})),n}(r[0])),a=0,u=0;for(;u<r[0];){if(0===n[a][u].num)for(let t=a+1;t<r[0];t++)0!==n[t][u].num&&(y(n,a,t),y(i,a,t));if(0!==n[a][u].num){if(1!==n[a][u].num||1!==n[a][u].den){let t=e(n[a][u].num,n[a][u].den);for(let e=0;e<r[1];e++)n[a][e]=n[a][e].div(t),i[a][e]=i[a][e].div(t)}for(let t=a+1;t<r[0];t++){let e=n[t][u];for(let o=0;o<r[1];o++)n[t][o]=n[t][o].sub(e.mul(n[a][o])),i[t][o]=i[t][o].sub(e.mul(i[a][o]))}}a+=1,u+=1}let s=r[0]-1;if(1!==n[s][s].num||1!==n[s][s].den){let t=e(n[s][s].num,n[s][s].den);for(let e=0;e<r[1];e++)n[s][e]=n[s][e].div(t),i[s][e]=i[s][e].div(t)}for(let t=r[0]-1;t>0;t--)for(let o=t-1;o>=0;o--){let a=e(-n[o][t].num,n[o][t].den);for(let e=0;e<r[1];e++)n[o][e]=a.mul(n[t][e]).add(n[o][e]),i[o][e]=a.mul(i[t][e]).add(i[o][e])}return function(t){let n=[];return t.forEach(((t,r)=>{n.push(t.map((t=>t.num/t.den)))})),n}(i)}(t),merge:i(t),map:n=>function(t,n){const r=o(t),e=[];for(let i=0;i<r[0];i++)if(Array.isArray(t[i])){e[i]=[];for(let o=0;o<r[1];o++)e[i][o]=n(t[i][o],[i,o],t)}else e[i]=n(t[i],[i,0],t);return e}(t,n),equals:n=>function(t,n){let r=t,e=n(),i=o(r),a=o(e);return!!i.every(((t,n)=>t===a[n]))&&r.every(((t,n)=>t.every(((t,r)=>Math.abs(t-e[n][r])<1e-10))))}(t,n)}}(t))}},276:t=>{t.exports=function(t){return{top:n=>function(t,n){if((t[0].length||t.length)!==(n[n.length-1].length||n.length))return t;Array.isArray(t[0])||(t=[t]),Array.isArray(n[n.length-1])||(n=[n]);for(let r=n.length-1;r>=0;r--)t.unshift(n[r].map((t=>t)));return t}(t,n),bottom:n=>function(t,n){if((t[t.length-1].length||t.length)!==(n[0].length||n.length))return t;Array.isArray(t[t.length-1])||(t=[t]),Array.isArray(n[0])||(n=[n]);for(let r=0;r<n.length;r++)t.push(n[r].map((t=>t)));return t}(t,n),left:n=>function(t,n){let r=t.length,e=n.length;if(!Array.isArray(t[0])&&!Array.isArray(n[0]))return t.unshift.apply(t,n),t;if(r!==e)return t;for(let e=0;e<r;e++)t[e].unshift.apply(t[e],n[e].map((t=>t)));return t}(t,n),right:n=>function(t,n){let r=t.length,e=n.length;if(!Array.isArray(t[0])&&!Array.isArray(n[0]))return t.push.apply(t,n),t;if(r!==e)return t;for(let e=0;e<r;e++)t[e].push.apply(t[e],n[e].map((t=>t)));return t}(t,n)}}},673:t=>{function n(t,e){return e=e||1,-1===Math.sign(e)&&(t=-t,e=-e),{num:t,den:e,add:r=>n(t*r.den+e*r.num,e*r.den),sub:r=>n(t*r.den-e*r.num,e*r.den),mul:n=>r(n,t,e),div:i=>r(n(i.den,i.num),t,e)}}function r(t,r,e){let i=Math.sign(r)*Math.sign(t.num),o=Math.sign(e)*Math.sign(t.den);return Math.abs(r)===Math.abs(t.den)&&Math.abs(e)===Math.abs(t.num)||(Math.abs(e)===Math.abs(t.num)?(i*=Math.abs(r),o*=Math.abs(t.den)):Math.abs(r)===Math.abs(t.den)?(i*=Math.abs(t.num),o*=Math.abs(e)):(i=r*t.num,o=e*t.den)),n(i,o)}t.exports=n}},n={};function r(e){var i=n[e];if(void 0!==i)return i.exports;var o=n[e]={exports:{}};return t[e](o,o.exports,r),o.exports}r.d=(t,n)=>{for(var e in n)r.o(n,e)&&!r.o(t,e)&&Object.defineProperty(t,e,{enumerable:!0,get:n[e]})},r.o=(t,n)=>Object.prototype.hasOwnProperty.call(t,n),r.r=t=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})};var e={};return(()=>{r.r(e),r.d(e,{ForwardLayer:()=>j,ForwardNetwork:()=>M,JeffFunction:()=>k,MaskingType:()=>t,StaticFunction:()=>w});var t,n=r(423);function i(t){return i="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},i(t)}function o(t){return function(t){if(Array.isArray(t))return c(t)}(t)||function(t){if("undefined"!=typeof Symbol&&null!=t[Symbol.iterator]||null!=t["@@iterator"])return Array.from(t)}(t)||s(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function a(t,n){var r="undefined"!=typeof Symbol&&t[Symbol.iterator]||t["@@iterator"];if(!r){if(Array.isArray(t)||(r=s(t))||n&&t&&"number"==typeof t.length){r&&(t=r);var e=0,i=function(){};return{s:i,n:function(){return e>=t.length?{done:!0}:{done:!1,value:t[e++]}},e:function(t){throw t},f:i}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}var o,a=!0,u=!1;return{s:function(){r=r.call(t)},n:function(){var t=r.next();return a=t.done,t},e:function(t){u=!0,o=t},f:function(){try{a||null==r.return||r.return()}finally{if(u)throw o}}}}function u(t,n){return function(t){if(Array.isArray(t))return t}(t)||function(t,n){var r=null==t?null:"undefined"!=typeof Symbol&&t[Symbol.iterator]||t["@@iterator"];if(null!=r){var e,i,o,a,u=[],s=!0,c=!1;try{if(o=(r=r.call(t)).next,0===n){if(Object(r)!==r)return;s=!1}else for(;!(s=(e=o.call(r)).done)&&(u.push(e.value),u.length!==n);s=!0);}catch(t){c=!0,i=t}finally{try{if(!s&&null!=r.return&&(a=r.return(),Object(a)!==a))return}finally{if(c)throw i}}return u}}(t,n)||s(t,n)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function s(t,n){if(t){if("string"==typeof t)return c(t,n);var r=Object.prototype.toString.call(t).slice(8,-1);return"Object"===r&&t.constructor&&(r=t.constructor.name),"Map"===r||"Set"===r?Array.from(t):"Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)?c(t,n):void 0}}function c(t,n){(null==n||n>t.length)&&(n=t.length);for(var r=0,e=new Array(n);r<n;r++)e[r]=t[r];return e}function l(t,n){if("function"!=typeof n&&null!==n)throw new TypeError("Super expression must either be null or a function");t.prototype=Object.create(n&&n.prototype,{constructor:{value:t,writable:!0,configurable:!0}}),Object.defineProperty(t,"prototype",{writable:!1}),n&&f(t,n)}function f(t,n){return f=Object.setPrototypeOf?Object.setPrototypeOf.bind():function(t,n){return t.__proto__=n,t},f(t,n)}function h(t){var n=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(t){return!1}}();return function(){var r,e=p(t);if(n){var o=p(this).constructor;r=Reflect.construct(e,arguments,o)}else r=e.apply(this,arguments);return function(t,n){if(n&&("object"===i(n)||"function"==typeof n))return n;if(void 0!==n)throw new TypeError("Derived constructors may only return object or undefined");return y(t)}(this,r)}}function y(t){if(void 0===t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return t}function p(t){return p=Object.setPrototypeOf?Object.getPrototypeOf.bind():function(t){return t.__proto__||Object.getPrototypeOf(t)},p(t)}function d(t,n){for(var r=0;r<n.length;r++){var e=n[r];e.enumerable=e.enumerable||!1,e.configurable=!0,"value"in e&&(e.writable=!0),Object.defineProperty(t,b(e.key),e)}}function v(t,n,r){return n&&d(t.prototype,n),r&&d(t,r),Object.defineProperty(t,"prototype",{writable:!1}),t}function m(t,n){if(!(t instanceof n))throw new TypeError("Cannot call a class as a function")}function g(t,n,r){return(n=b(n))in t?Object.defineProperty(t,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):t[n]=r,t}function b(t){var n=function(t,n){if("object"!==i(t)||null===t)return t;var r=t[Symbol.toPrimitive];if(void 0!==r){var e=r.call(t,"string");if("object"!==i(e))return e;throw new TypeError("@@toPrimitive must return a primitive value.")}return String(t)}(t);return"symbol"===i(n)?n:String(n)}!function(t){t[t.NoMasking=0]="NoMasking",t[t.StraightMasking=1]="StraightMasking"}(t||(t={}));var A=v((function t(){m(this,t),g(this,"useNormalization",!0),g(this,"threshold",void 0),g(this,"activationFunction",void 0),g(this,"derivativeFunction",void 0)})),w=function(t){l(r,t);var n=h(r);function r(){var t;m(this,r);for(var e=arguments.length,i=new Array(e),o=0;o<e;o++)i[o]=arguments[o];return g(y(t=n.call.apply(n,[this].concat(i))),"useNormalization",!1),g(y(t),"threshold",.5),g(y(t),"activationFunction",(function(t){return t})),g(y(t),"derivativeFunction",(function(t,n,r,e,i){return 0})),t}return v(r)}(A),k=function(t){l(r,t);var n=h(r);function r(){var t;m(this,r);for(var e=arguments.length,i=new Array(e),o=0;o<e;o++)i[o]=arguments[o];return g(y(t=n.call.apply(n,[this].concat(i))),"threshold",.5),g(y(t),"activationFunction",(function(t){return t>0?t:0})),g(y(t),"derivativeFunction",(function(n,r,e,i,o){return 2*(1/(1+Math.exp(-(Math.pow(e,2)-t.threshold)))-(o?0:1))*e*(e>=0?1:0)*n})),t}return v(r)}(A),M=function(){function r(){m(this,r),g(this,"layers",[]),g(this,"connections",{}),g(this,"masks",{}),g(this,"lossFunction",{}),g(this,"sleeping",!1)}return v(r,[{key:"assignId",value:function(t){t.id=this.layers.length,this.layers.push(t)}},{key:"createConnection",value:function(r,e,i){var o=arguments.length>3&&void 0!==arguments[3]?arguments[3]:function(t,n){return Math.random()},a=arguments.length>4&&void 0!==arguments[4]?arguments[4]:t.NoMasking,s="".concat(r.id,"-").concat(e.id);this.connections[s]=n(Array(e.size).fill(0).map((function(t,n){return Array(r.size).fill(0).map((function(t,r){return o(n,r)}))}))),this.masks[s]=n(Array(e.size).fill(0).map((function(){return Array(r.size).fill(0).map((function(){return 1}))}))),this.lossFunction[s]=i,a===t.StraightMasking&&(r.size!=e.size?console.error("layers must have the same size to use neuron to neuron masking. got sizes: ".concat(r.size," -> ").concat(e.size)):this.masks[s]=n(n(Array(r.size).fill(Array(r.size).fill(0))).map((function(t,n){var r=u(n,2),e=r[0];return r[1]==e?1:0})))),e.incommingConnections.push(r.id),r.outcommingConnections.push(e.id)}},{key:"forward",value:function(){var t=arguments.length>0&&void 0!==arguments[0]&&arguments[0];this.sleeping=t;var n,r=a(this.layers);try{for(r.s();!(n=r.n()).done;)n.value.forwardPass()}catch(t){r.e(t)}finally{r.f()}var e,i=a(this.layers);try{for(i.s();!(e=i.n()).done;)e.value.adjusteWeights()}catch(t){i.e(t)}finally{i.f()}var o,u=a(this.layers);try{for(u.s();!(o=u.n()).done;)o.value.sync()}catch(t){u.e(t)}finally{u.f()}}},{key:"clear",value:function(){var t,n=a(this.layers);try{for(n.s();!(t=n.n()).done;){var r=t.value;r.setActivations(Array(r.size).fill(0))}}catch(t){n.e(t)}finally{n.f()}}},{key:"compactPurkinjeWeights",value:function(t,r){return n(n(this.connections[r].mul(this.masks[r])).prod(n(this.connections[t].mul(this.masks[t]))))}}]),r}();g(M,"Matrix",n);var j=function(){function r(t,e,i,o){m(this,r),g(this,"activations",void 0),g(this,"tempActivations",void 0),g(this,"size",void 0),g(this,"network",void 0),g(this,"id",void 0),g(this,"learningRate",.1),g(this,"shape",[]),g(this,"activationThreshold",2),g(this,"incommingConnections",[]),g(this,"outcommingConnections",[]),g(this,"lossFunction",void 0),g(this,"painLayer",!1),this.network=t,this.network.assignId(this),this.size=e,this.lossFunction=i,this.activations=n(Array(e).fill(0).map((function(){return[0]}))),this.tempActivations=n(Array(e).fill(0).map((function(){return[0]}))),this.shape=null!=o?o:[e]}return v(r,[{key:"connectToNextLayer",value:function(n,r){var e=arguments.length>2&&void 0!==arguments[2]?arguments[2]:function(t,n){return Math.random()},i=arguments.length>3&&void 0!==arguments[3]?arguments[3]:t.NoMasking;this.network.createConnection(this,n,r,e,i)}},{key:"forwardPass",value:function(){var t=this;this.tempActivations=n(Array(this.size).fill(0).map((function(){return[0]})));var r,e=a(this.incommingConnections.map((function(n){return t.network.layers[n]})));try{var i,o=function(){if(!(i=r.value).painLayer){var e=i.activations;if(t.lossFunction.useNormalization){var o=e().reduce((function(t,n){return t+Math.pow(n[0],2)}),0);o=0==o?1:o,e=n(e.map((function(t){return Math.pow(t,2)/o})))}t.tempActivations=n(t.tempActivations.add(n(n(t.network.connections["".concat(i.id,"-").concat(t.id)].mul(t.network.masks["".concat(i.id,"-").concat(t.id)])).prod(e))))}};for(e.s();!(r=e.n()).done;)o()}catch(t){e.e(t)}finally{e.f()}}},{key:"adjusteWeights",value:function(){for(var t=this,r=0,e=function(){var e=t.network.layers[t.incommingConnections[i]];if(e.painLayer)r=e.activations()[0][0];else{var o,a=null!==(o=t.network.lossFunction["".concat(e.id,"-").concat(t.id)])&&void 0!==o?o:e.lossFunction,s=n(n(Array(t.size).fill((a.useNormalization?e.getNormalizedActivations():e.activations)().map((function(t){return t[0]})))).map((function(n,r){var i=u(r,2),o=i[0],a=i[1];return[n,t.network.connections["".concat(e.id,"-").concat(t.id)](o,a)]}))),c=n(s.map((function(n,e){var i=u(n,2),o=i[0],s=i[1];return t.learningRate*a.derivativeFunction(o,s,t.tempActivations()[e[0]],a.activationFunction(t.tempActivations()[e[0]]),t.network.sleeping,r)})));t.network.connections["".concat(e.id,"-").concat(t.id)]=n(t.network.connections["".concat(e.id,"-").concat(t.id)].sub(c))}},i=0;i<this.incommingConnections.length;i++)e()}},{key:"getNormalizedActivations",value:function(){var t=this.activations,r=t().reduce((function(t,n){return t+Math.pow(n[0],2)}),0);return r=0==r?1:r,n(t.map((function(t){return Math.pow(t,2)/r})))}},{key:"sync",value:function(){this.tempActivations=n(this.tempActivations.map(this.lossFunction.activationFunction)),this.activations=n(o(this.tempActivations()).map((function(t){return o(t)})))}},{key:"setActivations",value:function(t){this.activations=n(t.map((function(t){return[t]})))}}]),r}()})(),e})()));