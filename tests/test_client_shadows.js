const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

global.window = {
  addEventListener() {},
  cancelAnimationFrame() {},
  requestAnimationFrame() { return 0; }
};
global.document = {};

const appSource = fs.readFileSync('frontend/app.js', 'utf8');
vm.runInThisContext(`${appSource}\nglobalThis.__sunfinderTest = { state, localSolarPosition, createClientShadows, tileBuildingsForVisibleMap };`);

const solar = globalThis.__sunfinderTest.localSolarPosition(new Date('2026-07-19T09:00:00Z'));
assert.equal(solar.altitude, 47.95928);
assert.equal(solar.azimuth, 149.1078);

const buildings = [{
  type: 'Feature',
  properties: { name: 'Test building', height: 15 },
  geometry: {
    type: 'Polygon',
    coordinates: [[[24.93, 60.16], [24.931, 60.16], [24.931, 60.161], [24.93, 60.16]]]
  }
}];
const shadows = globalThis.__sunfinderTest.createClientShadows(buildings, { altitude: 30, azimuth: 180 });

assert.equal(shadows.length, 1);
assert.equal(shadows[0].properties.length, 26);
assert.ok(shadows[0].geometry.coordinates[0].some(([, latitude]) => latitude > 60.161));

globalThis.__sunfinderTest.state.map = {
  getBounds() {
    return {
      getSouth: () => 60.16,
      getWest: () => 24.92,
      getNorth: () => 60.17,
      getEast: () => 24.94
    };
  },
  getCenter: () => ({ lat: 60.165 })
};
const tileBuildings = globalThis.__sunfinderTest.tileBuildingsForVisibleMap([{
  id: 91,
  properties: { render_height: 26 },
  geometry: {
    type: 'Polygon',
    coordinates: [[[24.93, 60.165], [24.931, 60.165], [24.931, 60.166], [24.93, 60.165]]]
  }
}]);

assert.equal(tileBuildings.length, 1);
assert.equal(tileBuildings[0].properties.height, 26);
assert.match(tileBuildings[0].properties.id, /^91:0:/);
