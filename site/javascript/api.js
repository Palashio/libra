$(function(){
    $(".dropdown-item").click(function(){
        $([document.documentElement, document.body]).animate({
            scrollTop: $("#" + $(this).attr("data")).offset().top - 115
        }, 500);
    });
});