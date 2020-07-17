if(screen.width < 1400){
   document.documentElement.style.setProperty("--zoom-scale", screen.width/1400);
}

$(function(){
    $(".dropdown-item").click(function(){
        $([document.documentElement, document.body]).stop();
        $([document.documentElement, document.body]).animate({
            scrollTop: ($("#" + $(this).attr("data")).position().top - 45)*getComputedStyle(document.documentElement).getPropertyValue('--zoom-scale')
        }, 500);
    });    
});

function color(index){ //indices start at 0
    var dds = document.getElementsByClassName("dropdown-item");
    var i;
    if(dds.length <= index){
        index=dds.length-1;
    }
    for(i = 0; i<dds.length; i++){
        if(i == index){
            document.getElementsByClassName("dropdown-item")[i].classList.add("selected"); //add selected to the given index and remove from everything else
        }
        else{
            document.getElementsByClassName("dropdown-item")[i].classList.remove("selected");
        }
    }
}

window.addEventListener('scroll', function(e) {
    var dds = document.getElementsByClassName("box");
    var i;
    for(i = 0; i<dds.length; i++){
        if((window.scrollY +50)/getComputedStyle(document.documentElement).getPropertyValue('--zoom-scale') < (dds[i].offsetHeight + dds[i].offsetTop)){
            color(i);
            break;
        }
    }
});