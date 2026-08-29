document.addEventListener("DOMContentLoaded", function() {

    const fileInput = document.getElementById('myFile');

    function previewImages(event) {
        const files = Array.from(event.target.files);
        const preview = document.getElementById('imagePreview');
        const selectBtn = document.querySelector('label[for="myFile"]');
        const analyseBtn = document.getElementById('analyseBtn');
        const changeBtn  = document.getElementById('changeBtn');
        const fileList   = document.getElementById('fileList');

        if (files.length === 0) {
            resetPreview();
            return;
        }

        // Show the first file as the large preview thumbnail
        const firstFile = files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
            const buffer = e.target.result;
            try {
                const ifds = UTIF.decode(buffer);
                UTIF.decodeImage(buffer, ifds[0]);
                const rgba = UTIF.toRGBA8(ifds[0]);
                const canvas = document.createElement('canvas');
                canvas.width  = ifds[0].width;
                canvas.height = ifds[0].height;
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(canvas.width, canvas.height);
                imageData.data.set(rgba);
                ctx.putImageData(imageData, 0, 0);
                const jpegURL = canvas.toDataURL('image/jpeg');
                preview.innerHTML = `<img src="${jpegURL}" alt="Uploaded Image" style="width:100%; height:100%; object-fit:contain;">`;
            } catch (err) {
                // Fallback for non-TIFF files
                const url = URL.createObjectURL(firstFile);
                preview.innerHTML = `<img src="${url}" alt="Uploaded Image" style="width:100%; height:100%; object-fit:contain;">`;
            }
        };
        reader.readAsArrayBuffer(firstFile);

        // Build the queue list
        if (fileList) {
            fileList.style.display = 'block';
            fileList.innerHTML = '';
            files.forEach((f, i) => {
                const li = document.createElement('li');
                li.innerHTML = `<span>${f.name}</span><span class="file-status"></span>`;
                fileList.appendChild(li);
            });
        }

        // Update the analyse button label
        if (analyseBtn) {
            analyseBtn.style.display  = 'inline-block';
            analyseBtn.textContent    = files.length === 1 ? 'Analyse Image' : `Analyse ${files.length} Images`;
        }
        if (changeBtn)  changeBtn.style.display  = 'inline-block';
        if (selectBtn)  selectBtn.style.display  = 'none';
    }

    function resetPreview() {
        const preview    = document.getElementById('imagePreview');
        const selectBtn  = document.querySelector('label[for="myFile"]');
        const analyseBtn = document.getElementById('analyseBtn');
        const changeBtn  = document.getElementById('changeBtn');
        const fileList   = document.getElementById('fileList');

        if (preview)  preview.innerHTML = '<span>Please upload one or more images!</span>';
        if (fileList) { fileList.style.display = 'none'; fileList.innerHTML = ''; }
        if (analyseBtn) analyseBtn.style.display = 'none';
        if (changeBtn)  changeBtn.style.display  = 'none';
        if (selectBtn)  selectBtn.style.display  = 'inline-block';
    }

    if (fileInput) {
        fileInput.addEventListener('change', previewImages);
    }

    // Unselect images
    const changeBtn = document.getElementById('changeBtn');
    if (changeBtn) {
        changeBtn.addEventListener('click', function() {
            if (fileInput) fileInput.value = '';
            resetPreview();
        });
    }

    //Compare functionality
    const compareBtn              = document.getElementById("compareBtn");
    const compareActionContainer  = document.getElementById("compareActionContainer");
    const checkboxes              = document.querySelectorAll(".compare-checkbox");

    if (compareBtn) {
        let compareMode = false;

        compareBtn.addEventListener("click", function () {
            compareMode = !compareMode;

            checkboxes.forEach(cb => {
                cb.style.display = compareMode ? "block" : "none";
                cb.checked = false;
            });

            compareBtn.textContent = compareMode ? "Cancel Compare" : "Compare Two Images";
            if (compareActionContainer) compareActionContainer.style.display = "none";
        });
    }

    checkboxes.forEach(cb => {
        cb.addEventListener("change", function () {
            const checked = document.querySelectorAll(".compare-checkbox:checked");
            if (checked.length > 2) {
                this.checked = false;
                alert("You can only compare two images.");
            }
            if (compareActionContainer) {
                compareActionContainer.style.display = (checked.length === 2) ? "block" : "none";
            }
        });
    });

    const compareImagesBtn = document.getElementById("compareImagesBtn");
    if (compareImagesBtn) {
        compareImagesBtn.addEventListener("click", function () {
            const checked = document.querySelectorAll(".compare-checkbox:checked");
            if (checked.length === 2) {
                const id1 = checked[0].dataset.id;
                const id2 = checked[1].dataset.id;
                window.location.href = `/compare?id1=${id1}&id2=${id2}`;
            }
        });
    }

    const compareODAImagesBtn = document.getElementById("compareODAImagesBtn");
    if (compareODAImagesBtn) {
        compareODAImagesBtn.addEventListener("click", function () {
            const checked = document.querySelectorAll(".compare-checkbox:checked");
            if (checked.length === 2) {
                const id1 = checked[0].dataset.id;
                const id2 = checked[1].dataset.id;
                window.location.href = `/compare_oda?id1=${id1}&id2=${id2}`;
            }
        });
    }

    //Filter functionality
    const filterBtn     = document.getElementById("filterBtn");
    const filterOptions = document.getElementById("filterOptions");
    if (filterBtn && filterOptions) {
        filterBtn.addEventListener("click", function() {
            filterOptions.style.display = filterOptions.style.display === "none" ? "block" : "none";
        });
    }

    const selectBtn             = document.getElementById("selectBtn");
    const selectActionContainer = document.getElementById("selectActionContainer");
    const scheckboxes           = document.querySelectorAll(".select-checkbox");

    if (selectBtn) {
        let selectMode = false;

        selectBtn.addEventListener("click", function () {
            selectMode = !selectMode;

            scheckboxes.forEach(cb => {
                cb.style.display = selectMode ? "block" : "none";
                cb.checked = false;
            });

            selectBtn.textContent = selectMode ? "Cancel Select" : "Select Images";
            if (!selectMode && selectActionContainer) {
                selectActionContainer.style.display = "none";
            }
        });
    }

    scheckboxes.forEach(cb => {
        cb.addEventListener("change", function () {
            const checked = document.querySelectorAll(".select-checkbox:checked");
            if (selectActionContainer) {
                selectActionContainer.style.display = (checked.length > 0) ? "block" : "none";
            }
        });
    });

    function getSelectedIds() {
        const checked = document.querySelectorAll(".select-checkbox:checked");
        return Array.from(checked).map(cb => cb.dataset.id).join(",");
    }

    const selectImagesBtn = document.getElementById("selectImagesBtn");
    if (selectImagesBtn) {
        selectImagesBtn.addEventListener("click", function () {
            const checked = document.querySelectorAll(".select-checkbox:checked");
            if (checked.length > 0) {
                const ids = Array.from(checked).map(cb => cb.dataset.id).join(",");
                window.location.href = `/select?ids=${ids}`;
            } else {
                alert("Please select at least one image.");
            }
        });
    }

    const viewTimespanBtn = document.getElementById("viewTimespanBtn");
    if (viewTimespanBtn) {
        viewTimespanBtn.addEventListener("click", function () {
            const ids = getSelectedIds();
            if (ids) window.location.href = `/timespan?ids=${ids}`;
        });
    }
 
    const exportCsvBtn = document.getElementById("exportCsvBtn");
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener("click", function () {
            const ids = getSelectedIds();
            if (ids) window.location.href = `/export?ids=${ids}`;
        });
    }

    const exportODACsvBtn = document.getElementById("exportODACsvBtn");
    if (exportODACsvBtn) {
        exportODACsvBtn.addEventListener("click", function () {
            const ids = getSelectedIds();
            if (ids) window.location.href = `/export_oda?ids=${ids}`;
        });
    }

    const deleteSelectedForm = document.getElementById("deleteSelectedForm");
    if (deleteSelectedForm) {
        deleteSelectedForm.addEventListener("submit", function(e) {
            e.preventDefault();
            const ids = getSelectedIds();
            if (!ids) return;
            if (!confirm("Are you sure you want to do this? This cannot be undone.")) return;
            document.getElementById("deleteIdsInput").value = ids;
            this.submit();
        });
    }

    const makePublicForm = document.getElementById("makePublicForm");
    if (makePublicForm) {
        makePublicForm.addEventListener("submit", function(e) {
            e.preventDefault();
            const ids = getSelectedIds();
            if (!ids) return;
            if (!confirm("Are you sure you want to do this?")) return;
            document.getElementById("publicIdsInput").value = ids;
            this.submit();
        });
    }

    const makePrivateForm = document.getElementById("makePrivateForm");
    if (makePrivateForm) {
        makePrivateForm.addEventListener("submit", function(e) {
            e.preventDefault();
            const ids = getSelectedIds();
            if (!ids) return;
            if (!confirm("Are you sure you want to do this?")) return;
            document.getElementById("privateIdsInput").value = ids;
            this.submit();
        });
    }

    //Functionality for country list dropdown and selection
    const countrySelect       = document.getElementById("filterCountry");
    const selectedContainer   = document.getElementById("selectedCountriesContainer");
    let selectedCountries     = [];

    function updateSelectedCountriesUI() {
        if (!selectedContainer) return;
        selectedContainer.innerHTML = "";
        selectedCountries.forEach((country, index) => {
            const div = document.createElement("div");
            div.style.cssText = "display:flex;justify-content:center;align-items:center;margin-bottom:5px;color:white;";
            div.innerHTML = `
                <span style="margin-right:5px;">${country}</span>
                <span style="cursor:pointer;color:white;font-weight:bold;" data-index="${index}">✖</span>
            `;
            selectedContainer.appendChild(div);
            div.querySelector("span[data-index]").addEventListener("click", function() {
                selectedCountries.splice(index, 1);
                updateSelectedCountriesUI();
            });
        });
    }

    if (countrySelect) {
        countrySelect.addEventListener("change", function() {
            const country = this.value;
            if (country && !selectedCountries.includes(country)) {
                selectedCountries.push(country);
                updateSelectedCountriesUI();
            }
            this.selectedIndex = 0;
        });
    }

    const partSelect   = document.getElementById('filterpart');
    const pcontainer   = document.getElementById('selectedPartsContainer');
    const phiddenInput = document.getElementById('selectedPartsInput');
    let selectedParts  = [];

    if (partSelect) {
        partSelect.addEventListener('change', function() {
            const val = this.value;
            if (val && !selectedParts.includes(val)) {
                selectedParts.push(val);
                updatePartTags();
            }
        });
    }

    function updatePartTags() {
        if (!pcontainer) return;
        pcontainer.innerHTML = '';
        selectedParts.forEach(part => {
            const tag = document.createElement('span');
            tag.style.cssText = "background:#ddd;padding:5px;margin:5px;border-radius:3px;display:inline-block;";
            tag.innerHTML = `${part} <span style="cursor:pointer;color:red" onclick="removePart('${part}')">&times;</span>`;
            pcontainer.appendChild(tag);
        });
        if (phiddenInput) phiddenInput.value = JSON.stringify(selectedParts);
    }

    window.removePart = function(part) {
        selectedParts = selectedParts.filter(p => p !== part);
        updatePartTags();
    };

    //Filter functionality
    const applyFiltersBtn  = document.getElementById("applyFilters");
    const removeFiltersBtn = document.getElementById("removeFilters");
    const contentContainers = Array.from(document.getElementsByClassName("content-container"));

    if (applyFiltersBtn) {
        applyFiltersBtn.addEventListener("click", function() {
            const dateFrom = document.getElementById("filterDateFrom")?.value;
            const dateTo   = document.getElementById("filterDateTo")?.value;

            contentContainers.forEach(container => {
                const row        = container.querySelector(".content-row");
                const rowCountry = row.dataset.country;
                const rowDate    = row.dataset.date;
                let show = true;

                if (selectedCountries.length > 0 && !selectedCountries.includes(rowCountry)) show = false;
                if (dateFrom && rowDate < dateFrom) show = false;
                if (dateTo   && rowDate > dateTo)   show = false;

                container.style.display = show ? "block" : "none";
            });

            if (removeFiltersBtn) removeFiltersBtn.style.display = "inline-block";
        });
    }

    if (removeFiltersBtn) {
        removeFiltersBtn.addEventListener("click", function() {
            contentContainers.forEach(c => c.style.display = "block");
            document.getElementById("filterDateFrom").value = "";
            document.getElementById("filterDateTo").value   = "";
            selectedCountries = [];
            updateSelectedCountriesUI();
            this.style.display = "none";
        });
    }

});
