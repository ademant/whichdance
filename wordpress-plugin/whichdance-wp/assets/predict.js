(function () {
  document.querySelectorAll('.whichdance-widget').forEach(function (widget) {
    var fileInput = widget.querySelector('.whichdance-file');
    var resultEl = widget.querySelector('.whichdance-result');

    fileInput.addEventListener('change', function () {
      var file = fileInput.files[0];
      if (!file) return;
      resultEl.textContent = 'analyzing…';

      var form = new FormData();
      form.append('file', file);

      fetch(whichdanceConfig.restUrl, {
        method: 'POST',
        headers: { 'X-WP-Nonce': whichdanceConfig.nonce },
        body: form,
      })
        .then(function (resp) {
          return resp.json().then(function (data) {
            if (!resp.ok) throw new Error(data.message || 'Request failed');
            return data;
          });
        })
        .then(render)
        .catch(function (err) {
          resultEl.textContent = 'Error: ' + err.message;
        });

      function render(data) {
        // Dance labels ultimately come from Funkwhale playlist names
        // (untrusted input), so build DOM nodes rather than interpolating
        // them into innerHTML.
        resultEl.textContent = '';
        var maxProb = Math.max.apply(null, data.top_labels.map(function (t) { return t[1]; }));

        data.top_labels.forEach(function (t) {
          var label = t[0], prob = t[1];
          var pct = Math.round((prob / maxProb) * 100);

          var row = document.createElement('div');
          row.className = 'whichdance-bar-row';

          var labelEl = document.createElement('div');
          labelEl.className = 'whichdance-bar-label';
          labelEl.textContent = label;

          var track = document.createElement('div');
          track.className = 'whichdance-bar-track';
          var fill = document.createElement('div');
          fill.className = 'whichdance-bar-fill';
          fill.style.width = pct + '%';
          track.appendChild(fill);

          var pctEl = document.createElement('div');
          pctEl.textContent = Math.round(prob * 100) + '%';

          row.appendChild(labelEl);
          row.appendChild(track);
          row.appendChild(pctEl);
          resultEl.appendChild(row);
        });

        var meta = document.createElement('p');
        meta.className = 'whichdance-meta';
        meta.textContent = Math.round(data.bpm) + ' bpm · ' + Math.round(data.duration_seconds) + 's';
        resultEl.appendChild(meta);
      }
    });
  });
})();
