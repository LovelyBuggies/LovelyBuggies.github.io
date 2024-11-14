
function rlSortList(sortBy) {
    const list = document.getElementById("rl-post-list");
    const items = Array.from(list.getElementsByTagName("li"));

    items.sort((a, b) => {
        if (sortBy === "date") {
            const dateA = new Date(a.dataset.date);
            const dateB = new Date(b.dataset.date);
            return dateB - dateA; // Decending order by date
        } else if (sortBy === "depth") {
            const depthA = parseInt(a.dataset.depth);
            const depthB = parseInt(b.dataset.depth);
            return depthA - depthB; // Ascending order by depth
        }
    });

    // Clear the list and append sorted items
    list.innerHTML = "";
    items.forEach(item => list.appendChild(item));
}


function marlSortList(sortBy) {
    const list = document.getElementById("marl-post-list");
    const items = Array.from(list.getElementsByTagName("li"));

    items.sort((a, b) => {
        if (sortBy === "date") {
            const dateA = new Date(a.dataset.date);
            const dateB = new Date(b.dataset.date);
            return dateB - dateA; // Decending order by date
        } else if (sortBy === "depth") {
            const depthA = parseInt(a.dataset.depth);
            const depthB = parseInt(b.dataset.depth);
            return depthA - depthB; // Ascending order by depth
        }
    });

    // Clear the list and append sorted items
    list.innerHTML = "";
    items.forEach(item => list.appendChild(item));
}


