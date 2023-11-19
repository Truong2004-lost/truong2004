// Your JavaScript code goes here
$(document).ready(function () {
    $("#loadDataBtn").click(function () {
        // Simulate loading data
        var data = "<p>Data loaded successfully!</p>";
        $("#dataContainer").html(data);
    });
});