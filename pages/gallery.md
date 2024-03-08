---
layout: base
title: 
subtitle: 
---
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.3/css/all.css">
<style>
/* carousel */
.carousel{
   position: relative;
}

.carousel_inner{
   width: 100%;
   overflow: hidden;
}

.carousel_inner::after{
   content: "";
   display: block;
   clear: both;
}

.carousel_item{
   position: relative;
   float: left;
   display: none;
   width: 100%;
   margin-right: -100%;
}

/* slide effect */
.carousel_item__active,
.carousel_item__pos_prev,
.carousel_item__pos_next{
   display: block;
}

.carousel_item__pos_prev{
   left: -100%;
}

.carousel_item__pos_next{
   left: 100%;
}

.carousel_item__prev{
   transform: translateX(100%);
   transition: transform .5s ease-in-out;
}

.carousel_item__next{
   transform: translateX(-100%);
   transition: transform .5s ease-in-out;
}

/* fade effect */
.carousel__fade .carousel_item__pos_prev,
.carousel__fade .carousel_item__pos_next{
   left: 0;
   opacity: 0;
}

.carousel__fade .carousel_item__prev,
.carousel__fade .carousel_item__next{
   transform: none;
   opacity: 1;
   transition: opacity .5s ease-in-out;
   
}

.carousel__fade .carousel_item__active.carousel_item__prev,
.carousel__fade .carousel_item__active.carousel_item__next{
   opacity: 0;
}

/* carousel */
.carousel_img{
   display: block;
   width: 80%;
   margin-left: auto;
    margin-right: auto;

}

.carousel_caption{
   position: absolute;
   top: 0;
   left: 0;
   display: flex;
   flex-direction: column;
   justify-content: center;
   width: 100%;
   height: 100%;
   padding: 3rem 4rem;
   text-align: center;
   color: #fff;
   z-index: 1;
}

.carousel_title{
   font-family: "Montserrat", sans-serif;
   font-weight: 600;
}

.carousel_description{
   margin-top: .75rem;
   line-height: 150%;
}

.carousel_indicator{
   position: absolute;
   bottom: 1rem;
   left: 50%;
   transform: translateX(-50%);
   display: flex;
   flex-direction: row;
   gap: .5rem;
   z-index: 1;
}

.carousel_dot{
   display: block;
   padding: .25rem;
   background-color: rgba(255, 255, 255, 1);
   border: none;
   border-radius: 50%;
   cursor: pointer;
   transition: background-color .5s ease-in-out;
}

.carousel_dot__active{
   background-color: lightskyblue;
   cursor: default;
   pointer-events: none;
}

.carousel_control{
   position: absolute;
   top: 20;
   left: 20;
   display: flex;
   flex-direction: row;
   justify-content: space-between;
   width: 100%;
   height: 100%;
}

.carousel_button{
   display: block;
   padding-inline: 1rem;
   font-family: "Raleway", sans-serif;
   font-size: 1.375rem;
   background-color: transparent;
   color: rgba(255, 255, 255, .25);
   border: none;
   cursor: pointer;
   transition: color .25s;
   z-index: 1;
}

.carousel_button:hover{
   color: rgba(135, 206, 250, .75);
}
</style>

<script>
  let onSlide = false;

window.addEventListener("load", () => {
   autoSlide();

   const dots = document.querySelectorAll(".carousel_dot");
   for (let i = 0; i < dots.length; i++) {
      dots[i].addEventListener("click", () => slide(i));
   }

   const buttonPrev = document.querySelector(".carousel_button__prev");
   const buttonNext = document.querySelector(".carousel_button__next");
   buttonPrev.addEventListener("click", () => slide(getItemActiveIndex() - 1));
   buttonNext.addEventListener("click", () => slide(getItemActiveIndex() + 1));
})

function autoSlide() {
   setInterval(() => {
      slide(getItemActiveIndex() + 1);
   }, 10000); // slide speed = 3s
}

function slide(toIndex) {
   if (onSlide)
      return;
   onSlide = true;

   const itemsArray = Array.from(document.querySelectorAll(".carousel_item"));
   const itemActive = document.querySelector(".carousel_item__active");
   const itemActiveIndex = itemsArray.indexOf(itemActive);
   let newItemActive = null;

   if (toIndex > itemActiveIndex) {
      // check if toIndex exceeds the number of carousel items
      if (toIndex >= itemsArray.length) {
         toIndex = 0;
      }

      newItemActive = itemsArray[toIndex];

      // start transition
      newItemActive.classList.add("carousel_item__pos_next");
      setTimeout(() => {
         newItemActive.classList.add("carousel_item__next");
         itemActive.classList.add("carousel_item__next");
      }, 50);
   } else {
      // check if toIndex exceeds the number of carousel items
      if (toIndex < 0) {
         toIndex = itemsArray.length - 1;
      }

      newItemActive = itemsArray[toIndex];

      // start transition
      newItemActive.classList.add("carousel_item__pos_prev");
      setTimeout(() => {
         newItemActive.classList.add("carousel_item__prev");
         itemActive.classList.add("carousel_item__prev");
      }, 20);
   }

   // remove all transition class and switch active class
   newItemActive.addEventListener("transitionend", () => {
      itemActive.className = "carousel_item";
      newItemActive.className = "carousel_item carousel_item__active";
      onSlide = false;
   }, {
      once: true
   });

   slideIndicator(toIndex);
}

function getItemActiveIndex() {
   const itemsArray = Array.from(document.querySelectorAll(".carousel_item"));
   const itemActive = document.querySelector(".carousel_item__active");
   const itemActiveIndex = itemsArray.indexOf(itemActive);
   return itemActiveIndex;
}

function slideIndicator(toIndex) {
   const dots = document.querySelectorAll(".carousel_dot");
   const dotActive = document.querySelector(".carousel_dot__active");
   const newDotActive = dots[toIndex];

   dotActive.classList.remove("carousel_dot__active");
   newDotActive.classList.add("carousel_dot__active");
}
</script>

<div class="carousel">
      <div class="carousel_inner">
         <div class="carousel_item carousel_item__active">
            <img src="{{site.baseurl}}/assets/img/PXL_20230720_060406788.jpg" alt="" class="carousel_img">
            <div class="carousel_caption">
               <h1 class="carousel_title">My pictures </h1>
            </div>
         </div>
         <div class="carousel_item">
             <img src="{{site.baseurl}}/assets/img/PXL_20230720_060406788.jpg" alt="" class="carousel_img">
         </div>
      </div>

      <div class="carousel_indicator">
         <button class="carousel_dot carousel_dot__active"></button>
         <button class="carousel_dot"></button>
         <button class="carousel_dot"></button>
      </div>

      <div class="carousel_control">
         <button class="carousel_button carousel_button__prev">
            <i class="fas fa-chevron-left"></i>
         </button>
         <button class="carousel_button carousel_button__next">
            <i class="fas fa-chevron-right"></i>
         </button>
      </div>
   </div>