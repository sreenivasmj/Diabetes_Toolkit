document.addEventListener("DOMContentLoaded", function() {
    // Get the suggestions div
    var suggestionsDiv = document.getElementById("suggestions");

    // Get the prediction result text
    var resultText = "{{ result_text }}";

    // Clear any existing content
    suggestionsDiv.innerHTML = "";

    // Check the prediction result and add appropriate suggestions
    if (resultText === "This Person is Diabetic") {
        suggestionsDiv.innerHTML = "<p>Consider exploring diabetes medication options.</p>";
    } else if (resultText === "This Person is Not Diabetic") {
        suggestionsDiv.innerHTML = "<p>Focus on maintaining a healthy lifestyle and diet to prevent diabetes.</p>";
    }
});
