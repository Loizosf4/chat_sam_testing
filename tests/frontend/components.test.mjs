import test from "node:test";
import assert from "node:assert/strict";
import {detailsHtml,objectListHtml} from "../../frontend/components.js";

const object={object_id:"abc",display_name:"Desk <main>",semantic_label:"desk",approval_status:"unreviewed",geometry_confidence:.72,orientation_confidence:.8,placement_confidence:.65,quality_warnings:["touches edge"],mask_revision:"rev"};
test("semantic list renders selection, approval, confidence, and accessible visibility",()=>{const html=objectListHtml([{...object,color:"#fff"}],"abc",new Map([["abc",false]]));assert.match(html,/aria-selected="true"/);assert.match(html,/is-hidden/);assert.match(html,/72%/);assert.match(html,/aria-label="Toggle Desk &lt;main&gt; visibility"/);assert.doesNotMatch(html,/<main>/)});
test("details render label, warnings, approval controls, history, and immutable-save state",()=>{const html=detailsHtml(object,[{revision_id:"rev",revision_number:2,edit_operation:"manual",author:"reviewer",timestamp:"2026-01-01T00:00:00Z"}],"desk",true);assert.match(html,/touches edge/);assert.match(html,/Approve/);assert.match(html,/Save as new revision/);assert.match(html,/immutable revision/);assert.match(html,/current/)});
