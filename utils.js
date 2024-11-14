function rlSortList(sortBy) {
    sortList('rl-post-list', sortBy);
}

function marlSortList(sortBy) {
    sortList('marl-post-list', sortBy);
}

function sortList(listId, sortBy) {
    const list = document.getElementById(listId);
    const items = Array.from(list.getElementsByTagName("li"));

    items.sort((a, b) => {
        if (sortBy === "date") {
            const dateA = new Date(a.dataset.date);
            const dateB = new Date(b.dataset.date);
            return dateB - dateA;
        } else if (sortBy === "depth") {
            const depthA = parseInt(a.dataset.depth);
            const depthB = parseInt(b.dataset.depth);
            return depthA - depthB;
        }
    });

    list.innerHTML = "";
    items.forEach(item => list.appendChild(item));
}

let lastSelectedTag = null;
function showTaggedItems(tag) {
    const lists = ['rl-post-list', 'marl-post-list'];
    const filteredList = document.getElementById("filtered-list");

    if (lastSelectedTag === tag) {
        filteredList.innerHTML = ""; 
        lastSelectedTag = null; 
        return;
    }
    lastSelectedTag = tag;
    filteredList.innerHTML = "";

    lists.forEach(listId => {
        const list = document.getElementById(listId);
        const items = Array.from(list.getElementsByTagName("li"));

        items.forEach(item => {
            const tags = item.dataset.tag ? item.dataset.tag.split(',') : [];
            if (tags.includes(tag)) {
                const clone = item.cloneNode(true);
                filteredList.appendChild(clone);
            }
        });
    });
}

function addTagAndDateInfo() {
    const lists = document.querySelectorAll("#rl-post-list li, #marl-post-list li");

    lists.forEach(item => {
        const annotateElement = item.querySelector("annotate");
        const date = item.getAttribute("data-date");
        const depth = item.getAttribute("data-depth");
        const tags = item.getAttribute("data-tag") || "NULL";
        const infoElement = document.createElement("span");
        infoElement.style.fontSize = "0.8em"; 
        infoElement.style.color = "#888888";
        infoElement.style.fontStyle = "italic";
        infoElement.innerHTML = `Tags: ${tags}; Specialty: ${depth}; Last Update: ${date}.`;
        annotateElement.insertAdjacentHTML("afterend", "<br>");
        annotateElement.insertAdjacentElement("afterend", infoElement);
        annotateElement.insertAdjacentHTML("afterend", "</br>");
    });
}
