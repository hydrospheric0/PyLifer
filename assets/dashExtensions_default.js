window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng, context) {
                const p = feature.properties;
                const icon = L.divIcon({
                    className: '',
                    html: `<div style="
            background: ${p.colour};
            color: #fff;
            font-weight: bold;
            font-size: 11px;
            width: 26px; height: 26px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            border: 2px solid rgba(255,255,255,0.25);
            box-shadow: 0 2px 6px rgba(0,0,0,0.6);
            cursor: pointer;">${p.rank}</div>`,
                    iconSize: [26, 26],
                    iconAnchor: [13, 13],
                });
                return L.marker(latlng, {
                    icon
                });
            }

            ,
        function1: function(feature, layer, context) {
            const p = feature.properties;
            const url = p.loc_id ? `https://ebird.org/hotspot/${p.loc_id}` : null;
            const nameHtml = url ?
                `<a href="${url}" target="_blank" style="color:#58a6ff">${p.loc_name}</a>` :
                p.loc_name;
            layer.bindPopup(
                `<div style="font-family:sans-serif;min-width:180px">
          <b>#${p.rank}</b> ${nameHtml}<br/>
          <span style="color:#999">Score: ${p.score.toFixed(1)}
          &nbsp;|&nbsp; eBird spp: ${p.hs_spp ?? '—'}</span><br/>
          <span style="color:#aaa; font-size:0.85em">
            ${p.state || '—'} &nbsp;/&nbsp; ${p.county || '—'}
          </span>
        </div>`
            );
        }

    }
});