(function() {
    //The fly head parts with associated colours
    var CLASS_INFO = [
        { name: 'Right Eye',   colour: 'limegreen' },
        { name: 'Left Eye',  colour: 'yellow' },
        { name: 'Top Eye',    colour: 'tomato' },
        { name: 'Antenna',    colour: 'orange' },
        { name: 'Other',      colour: 'orchid' }
    ];
    function getInfo(cls) {
        return CLASS_INFO[cls] || CLASS_INFO[4];
    }
    var canvas   = document.getElementById('maincanvas');
    var ctx      = canvas.getContext('2d');
    var wrap     = document.getElementById('canvaswrap');
    var polylist = document.getElementById('polylist');
    var loader   = document.getElementById('loader');
    var toast    = document.getElementById('toast');
    var zoomlbl  = document.getElementById('zoomlabel');
    var fname    = document.getElementById('fname');
    var pcount   = document.getElementById('polycount');
    var infonone   = document.getElementById('infonone');
    var infodetail = document.getElementById('infodetail');
    var infoclass  = document.getElementById('infoclass');
    var infoconf   = document.getElementById('infoconf');
    var infopts    = document.getElementById('infopts');
    var btnreset  = document.getElementById('btnreset');
    var btnfinish = document.getElementById('btnfinish');

    //Editor state variables
    var polys      = [];
    var origpolys  = null;
    var history    = [];
    var selPoly    = null;
    var selPt      = null;
    var dragging   = false;
    var scale      = 1.0;
    var offX       = 0;
    var offY       = 0;
    var imgW       = 0;
    var imgH       = 0;
    var img        = null;
    var imageId    = null;
    var toastTimer = null;

    //Scroll wheel down panning state
    var panning  = false;
    var panStart = null;

    function deepcopy(x) {
        return JSON.parse(JSON.stringify(x));
    }

    function saveHistory() {
        history.push(deepcopy(polys));
        if (history.length > 50) history.shift();
    }

    function showToast(msg, isErr) {
        toast.textContent = msg;
        toast.className = 'show' + (isErr ? ' err' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function() { toast.className = ''; }, 2600);
    }

    //Fetch detection data from the flask session
    function init() {
        fetch('/get_detections_single')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.error) {
                    showToast(data.error, true);
                    loader.style.display = 'none';
                    return;
                }

                polys     = data.polygons || [];
                origpolys = deepcopy(polys);
                imageId   = data.image_id || null;
                fname.textContent = data.filename || 'image';
                updateCount();

                img = new Image();
                img.onload = function() {
                    imgW = img.naturalWidth;
                    imgH = img.naturalHeight;
                    canvas.width  = imgW;
                    canvas.height = imgH;
                    fitToWindow();
                    buildList();
                    draw();
                    loader.style.display = 'none';
                };
                img.onerror = function() {
                    showToast('Could not load image', true);
                    loader.style.display = 'none';
                };
                img.src = data.image_url;
            })
            .catch(function(e) {
                showToast('Failed to load: ' + e.message, true);
                loader.style.display = 'none';
            });
    }

    function updateCount() {
        var n = polys.length;
        pcount.textContent = n + ' polygon' + (n === 1 ? '' : 's');
    }

    function applyTransform() {
        canvas.style.transform = 'translate(' + offX + 'px, ' + offY + 'px) scale(' + scale + ')';
        zoomlbl.textContent = 'Zoom: ' + Math.round(scale * 100) + '%';
    }

    //This is to centre the image
    function fitToWindow() {
        var ww = wrap.clientWidth;
        var wh = wrap.clientHeight;
        var sx = ww / imgW;
        var sy = wh / imgH;
        scale = Math.min(sx, sy) * 0.95;
        offX = (ww - imgW * scale) / 2;
        offY = (wh - imgH * scale) / 2;
        applyTransform();
    }

    //Canvas mouse event to get coords
    function toImg(e) {
        var rect = canvas.getBoundingClientRect();
        return [
            (e.clientX - rect.left) / scale,
            (e.clientY - rect.top)  / scale
        ];
    }

    //Gets radius in the image pixels and shrinks when zoomed out
    function hitR() {
        return 12 / scale;
    }

    function findPoint(ix, iy) {
        var best = null;
        var bestd = hitR();
        for (var i = 0; i < polys.length; i++) {
            var pts = polys[i].points;
            for (var j = 0; j < pts.length; j++) {
                var d = Math.hypot(ix - pts[j][0], iy - pts[j][1]);
                if (d < bestd) {
                    bestd = d;
                    best  = { i: i, j: j };
                }
            }
        }
        return best;
    }

    function pointInPoly(pts, x, y) {
        var inside = false;
        for (var i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            var xi = pts[i][0], yi = pts[i][1];
            var xj = pts[j][0], yj = pts[j][1];
            if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
                inside = !inside;
        }
        return inside;
    }

    function findPolyAt(ix, iy) {
        //Iterates backwards for the last drawn polygon
        for (var i = polys.length - 1; i >= 0; i--) {
            if (pointInPoly(polys[i].points, ix, iy)) return i;
        }
        return null;
    }

    function distToSegment(px, py, x1, y1, x2, y2) {
        var dx = x2 - x1, dy = y2 - y1;
        var lensq = dx * dx + dy * dy;
        var t = lensq === 0 ? 0 : Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lensq));
        var cx = x1 + t * dx;
        var cy = y1 + t * dy;
        return { dist: Math.hypot(px - cx, py - cy), cx: cx, cy: cy };
    }

    function findEdge(ix, iy) {
        if (selPoly === null) return null;
        var pts   = polys[selPoly].points;
        var best  = null;
        var bestd = 20 / scale;
        for (var i = 0; i < pts.length; i++) {
            var next = (i + 1) % pts.length;
            var res  = distToSegment(ix, iy, pts[i][0], pts[i][1], pts[next][0], pts[next][1]);
            if (res.dist < bestd) {
                bestd = res.dist;
                best  = { idx: i, x: res.cx, y: res.cy };
            }
        }
        return best;
    }

    //Redraws everything from scratch each frame
    function draw() {
        if (!img) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        //Filled polygons first, then verteces points are drawn on top after
        for (var i = 0; i < polys.length; i++) {
            var p    = polys[i];
            var pts  = p.points;
            var info = getInfo(p.cls);
            if (pts.length < 3) continue;
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (var k = 1; k < pts.length; k++) {
                ctx.lineTo(pts[k][0], pts[k][1]);
            }
            ctx.closePath();
            ctx.globalAlpha = 0.25;
            ctx.fillStyle   = info.colour;
            ctx.fill();
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = info.colour;
            ctx.lineWidth   = i === selPoly ? 2.5 : 1.5;
            ctx.stroke();

            //Labels at centre of the polygons
            var cx = 0, cy = 0;
            for (var k = 0; k < pts.length; k++) { cx += pts[k][0]; cy += pts[k][1]; }
            cx /= pts.length;
            cy /= pts.length;
            var fs   = Math.max(10, Math.min(20, imgW / 55));
            var lab  = info.name + ' ' + p.conf.toFixed(2);
            ctx.font = 'bold ' + fs + 'px Arial';
            var tw   = ctx.measureText(lab).width;
            ctx.fillStyle = info.colour;
            ctx.fillRect(cx - 3, cy - fs - 2, tw + 8, fs + 6);
            ctx.fillStyle = 'black';
            ctx.fillText(lab, cx + 1, cy + 1);
        }
        //Vertex points
        for (var i = 0; i < polys.length; i++) {
            var pts  = polys[i].points;
            var info = getInfo(polys[i].cls);

            for (var j = 0; j < pts.length; j++) {
                var isSel = (i === selPoly && j === selPt);
                var r     = isSel ? 7 : 5;

                ctx.beginPath();
                ctx.arc(pts[j][0], pts[j][1], r, 0, Math.PI * 2);
                ctx.fillStyle   = isSel ? 'white' : (i === selPoly ? info.colour : 'rgba(255,255,255,0.7)');
                ctx.fill();
                ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                ctx.lineWidth   = 1.5;
                ctx.stroke();
            }
        }
    }

    //Canvas mouse events
    canvas.addEventListener('mousedown', function(e) {
        if (e.button !== 0) return;
        e.preventDefault();

        var coords = toImg(e);
        var ix = coords[0], iy = coords[1];
        var hit = findPoint(ix, iy);

        if (hit) {
            selPoly   = hit.i;
            selPt     = hit.j;
            dragging  = true;
            saveHistory();
        } else {
            var poly = findPolyAt(ix, iy);
            selPoly  = poly;
            selPt    = null;
            dragging = false;
        }

        updateInfo();
        buildList();
        draw();
    });

    canvas.addEventListener('mousemove', function(e) {
        if (!dragging || selPt === null) return;
        e.preventDefault();

        var coords = toImg(e);
        var ix = Math.max(0, Math.min(imgW - 1, coords[0]));
        var iy = Math.max(0, Math.min(imgH - 1, coords[1]));

        polys[selPoly].points[selPt] = [ix, iy];
        updateInfo();
        draw();
    });

    canvas.addEventListener('mouseup', function(e) {
        if (e.button === 0) dragging = false;
    });

    canvas.addEventListener('dblclick', function(e) {
        e.preventDefault();
        var coords = toImg(e);
        var edge   = findEdge(coords[0], coords[1]);

        if (!edge) {
            showToast('Select a polygon first, then double-click near an edge');
            return;
        }

        polys[selPoly].points.splice(edge.idx + 1, 0, [edge.x, edge.y]);
        saveHistory();
        selPt = edge.idx + 1;
        buildList();
        updateInfo();
        draw();
        showToast('Point added');
    });

    canvas.addEventListener('contextmenu', function(e) {
        e.preventDefault();
        var coords = toImg(e);
        var hit    = findPoint(coords[0], coords[1]);
        if (!hit) return;

        if (polys[hit.i].points.length <= 3) {
            showToast('Polygon needs at least 3 points', true);
            return;
        }
        saveHistory();
        polys[hit.i].points.splice(hit.j, 1);
        selPoly = hit.i;
        selPt   = null;
        buildList();
        updateInfo();
        draw();
        showToast('Point deleted');
    });

    canvas.addEventListener('wheel', function(e) {
        e.preventDefault();
        var rect   = wrap.getBoundingClientRect();
        var mx     = e.clientX - rect.left;
        var my     = e.clientY - rect.top;
        var factor = e.deltaY < 0 ? 1.1 : 0.9;
        var news   = Math.max(0.1, Math.min(10, scale * factor));

        offX   = mx - (mx - offX) * (news / scale);
        offY   = my - (my - offY) * (news / scale);
        scale  = news;
        applyTransform();
    }, { passive: false });

    //Scroll wheel down pan
    wrap.addEventListener('mousedown', function(e) {
        if (e.button !== 1) return;
        e.preventDefault();
        panning  = true;
        panStart = { x: e.clientX, y: e.clientY, ox: offX, oy: offY };
    });

    window.addEventListener('mousemove', function(e) {
        if (!panning) return;
        offX = panStart.ox + (e.clientX - panStart.x);
        offY = panStart.oy + (e.clientY - panStart.y);
        applyTransform();
    });

    window.addEventListener('mouseup', function(e) {
        if (e.button === 1) panning = false;
    });

    //Keyboard controls
    window.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            selPoly = null;
            selPt   = null;
            buildList();
            updateInfo();
            draw();
        }
        if (e.key === 'c' || e.key === 'C') {
            fitToWindow();
        }
        if (e.key === 'z' || e.key === 'Z') {
            if (history.length === 0) {
                showToast('Nothing to undo', true);
                return;
            }
            polys   = history.pop();
            selPoly = null;
            selPt   = null;
            buildList();
            updateInfo();
            draw();
            showToast('Undone');
        }
    });

    //The sidebar polygon list
    function buildList() {
        polylist.innerHTML = '';
        updateCount();
        for (var i = 0; i < polys.length; i++) {
            var p    = polys[i];
            var info = getInfo(p.cls);
            var div = document.createElement('div');
            div.className = 'polyitem' + (i === selPoly ? ' active' : '');
            div.innerHTML =
                '<div class="ptop">' +
                    '<div class="dot" style="background:' + info.colour + '"></div>' +
                    '<span class="pname" style="color:' + info.colour + '">' + info.name + '</span>' +
                    '<span class="pconf">' + (p.conf * 100).toFixed(0) + '%</span>' +
                '</div>' +
                '<div class="pmeta">' + p.points.length + ' pts</div>';

            //This closure is needed to get i correctly in loop
            (function(idx) {
                div.addEventListener('click', function() {
                    selPoly = idx;
                    selPt   = null;
                    buildList();
                    updateInfo();
                    draw();
                });
            })(i);

            polylist.appendChild(div);
        }
    }

    function updateInfo() {
        if (selPoly !== null && polys[selPoly]) {
            var p    = polys[selPoly];
            var info = getInfo(p.cls);
            infonone.style.display   = 'none';
            infodetail.style.display = 'block';
            infoclass.textContent    = info.name;
            infoclass.style.color    = info.colour;
            infoconf.textContent     = (p.conf * 100).toFixed(1) + '%';
            infopts.textContent      = p.points.length;
        } else {
            infonone.style.display   = 'block';
            infodetail.style.display = 'none';
        }
    }

    btnreset.addEventListener('click', function() {
        if (!confirm('Reset all polygons back to original detections?')) return;
        polys  = deepcopy(origpolys);
        selPoly = null;
        selPt   = null;
        buildList();
        updateInfo();
        draw();
        showToast('Reset to original');
    });

    btnfinish.addEventListener('click', function() {
        btnfinish.disabled   = true;
        btnfinish.textContent = 'Saving';
        fetch('/save_detections_single', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ polygons: polys, image_id: imageId })
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.success) {
                showToast('Saved and redirecting');
                setTimeout(function() {
                    window.location.href = data.redirect || '/resultsf';
                }, 1200);
            } else {
                throw new Error(data.error || 'Error');
            }
        })
        .catch(function(e) {
            showToast('Save failed: ' + e.message, true);
            btnfinish.disabled   = false;
            btnfinish.textContent = 'Finish & Save';
        });
    });

    init();

})();
